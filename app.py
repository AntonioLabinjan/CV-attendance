from imports import *


from db_startup import create_db
create_db()

from model_loader import load_clip_model

# Load-an stvari iz env fajla
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')

# Config za email api
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT'))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS') == 'True'
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')

# SendGrid API Key
sendgrid_api_key = os.getenv('SENDGRID_API_KEY')



# Init model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# In-memory storage za lica; ovo čak i ne triba spremat u bazu jer: 
                                    # a) svaki put pozovemo funkciju za loading usera kad palimo app i vjerojatno je isto brzo dali ih učita ovako, ili selektira iz baze
                                    # b) neman ni približnu ideju kako vraga se slike moru spremit u sql bazu...vjerojatno bi trebalo setupirat neki firebase samo za slike, ali dela i ovako...ako dela, ne tičen niš
known_face_encodings = []
known_face_names = []

# Track attendance for the current session
logged_names = set()


# ovo bi bilo dobro spremit u .env fajl, ali to ću ben delat
#app.secret_key = 'DO_NOT_VISIT_GRMIALDA' # RIJEŠENO


# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Model za usera
class User:
    def __init__(self, id, username, password, email):
        self.id = id
        self.username = username
        self.password = password
        self.email = email

    # Flask-Login needs these properties to work correctly
    @property
    def is_active(self):
        # Return True if the user is active. You can modify this based on user status.
        return True

    @property
    def is_authenticated(self):
        # Return True if the user is authenticated
        return True

    @property
    def is_anonymous(self):
        # Return False because this is not an anonymous user
        return False

    def get_id(self):
        # Return the user's ID as a string
        return str(self.id)


# In-memory user storage (can be replaced with a database)
# users = {} OVO MI VIŠE NE TRIBA

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = c.fetchone()
    conn.close()

    if user:
        return User(id=user[0], username=user[1], password=user[2], email=user[3])
    return None


mail = Mail()


app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'attendance.logged@gmail.com'  # Forši da napravin još 1 mail da bude malo više smisleno
app.config['MAIL_PASSWORD'] = 'ATTENDANCE/2025'  # Password
app.config['MAIL_DEFAULT_SENDER'] = 'attendance.logged@gmail.com'  # Adresa iz koje šaljen

mail = Mail(app)



# Function to add a known face
def add_known_face(image_path, name):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} not found.")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inputs = processor(images=image_rgb, return_tensors="pt")
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    embedding = outputs.cpu().numpy().flatten()
    known_face_encodings.append(embedding / np.linalg.norm(embedding))  # Normalize the embedding
    known_face_names.append(name)
    print(f"Added face for {name} from {image_path}")

# Load all known faces from the 'known_faces' directory
def load_known_faces():
    for student_name in os.listdir('known_faces'):
        student_dir = os.path.join('known_faces', student_name)
        if os.path.isdir(student_dir):
            for filename in os.listdir(student_dir):
                image_path = os.path.join(student_dir, filename)
                try:
                    add_known_face(image_path, student_name)
                except ValueError as e:
                    print(e)

    # Debugging
    print(f"Loaded {len(known_face_encodings)} known face encodings")
    print(f"Known face names: {known_face_names}")

# Load known faces at startup
load_known_faces()

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

# Face detection using Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')




# Inicijaliziranje atributa za trenutnu prisutnost
current_subject = None
attendance_date = None
start_time = None
end_time = None


# Password validation
def validate_password(password):
    if len(password) < 8:
        return "Password must be at least 8 characters long."
    if not re.search(r"[A-Z]", password):
        return "Password must contain at least one uppercase letter."
    if not re.search(r"[a-z]", password):
        return "Password must contain at least one lowercase letter."
    if not re.search(r"\d", password):
        return "Password must contain at least one number."
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return "Password must contain at least one special character."
    return None

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        repeat_password = request.form['repeat_password']
        email = request.form['email']

        # Check if passwords match
        if password != repeat_password:
            flash("Passwords do not match. Please try again.", "error")
            return redirect(url_for('signup'))

        # Validate password
        password_error = validate_password(password)
        if password_error:
            flash(password_error, "error")
            return redirect(url_for('signup'))

        # Check if username already exists
        try:
            conn = sqlite3.connect('attendance.db')
            c = conn.cursor()

            # Check if username or email already exists
            c.execute("SELECT * FROM users WHERE username = ? OR email = ?", (username, email))
            if c.fetchone() is not None:
                flash("Username or email already taken. Please choose a different one.", "error")
                conn.close()
                return redirect(url_for('signup'))

            # Hash the password
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

            # Insert new user into the database
            c.execute("INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
                      (username, hashed_password, email))
            conn.commit()
            conn.close()

            flash("Signup successful! Please log in.", "success")
            return redirect(url_for('login'))

        except sqlite3.IntegrityError:
            flash("An error occurred during signup. Please try again.", "error")
            return redirect(url_for('signup'))

    return render_template('signup.html')



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('attendance.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = c.fetchone()  # Fetches user row
        conn.close()

        if user:
            # User is found, verify the password
            user_id, db_username, db_password, db_email = user
            if check_password_hash(db_password, password):
                # Successful login
                login_user(User(id=user_id, username=db_username, password=db_password, email=db_email))
                return redirect(url_for('index'))

        flash("Invalid username or password")
        return redirect(url_for('login'))

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/set_subject', methods=['GET', 'POST'])
@login_required
def set_subject():
    global current_subject, attendance_date, start_time, end_time
    if request.method == 'POST':
        current_subject = request.form['subject']
        attendance_date = request.form['date']
        start_time = request.form['start_time']
        end_time = request.form['end_time']
        return render_template('set_subject_success.html', 
                               subject=current_subject, 
                               date=attendance_date, 
                               start_time=start_time, 
                               end_time=end_time)
    return render_template('set_subject.html')

def is_within_time_interval():
    now = datetime.now()
    current_time = now.strftime("%H:%M")
    current_date = now.strftime("%Y-%m-%d")
    return (current_date == attendance_date and 
            start_time <= current_time <= end_time)

@app.route('/add_student', methods=['GET', 'POST'])
@login_required
def add_student():
    if request.method == 'POST':
        name = request.form['name']
        images = request.files.getlist('images')
        
        # Novi subfolder za nove studente
        student_dir = os.path.join('known_faces', name)
        os.makedirs(student_dir, exist_ok=True)

        for image in images:
            # Add image in new subfolder
            image_path = os.path.join(student_dir, image.filename)
            image.save(image_path)
            add_known_face(image_path, name)
        
        return render_template('add_student_success.html', name=name)

    return render_template('add_student.html')

# Route to confirm success
@app.route('/add_student_success')
def add_student_success():
    return render_template('add_student_success.html')


def send_attendance_notification(name, date, time, subject):
    message = Mail(
        from_email='attendance.logged@gmail.com', 
        to_emails='alabinjan6@gmail.com', # napravit neki official mail za ovaj app da ne koristin svoj stari mail
        subject=f'Attendance Logged for {name}',
        plain_text_content=f'Attendance for {name} in {subject} on {date} at {time} was successfully logged.'
    )
    
    try:
        print("Attempting to send email...")
        sg = SendGridAPIClient('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')
        response = sg.send(message)
        print(f"Email sent: {response.status_code}")
        print(f"Response body: {response.body}")  # Debug
    except Exception as e:
        print(f"Error sending email: {str(e)}")



def log_attendance(name, frame):
    global current_subject, attendance_date, start_time, end_time
    if current_subject is None or not is_within_time_interval():
        print("Subject is not set or current time is outside of allowed interval. Attendance not logged.")
        return frame

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    # Create datetime object for start time
    start_time_obj = datetime.strptime(f"{attendance_date} {start_time}", "%Y-%m-%d %H:%M")

    # Toleriramo do 15 minuta kašnjenja (akademska četvrt :))
    late_time_obj = start_time_obj + timedelta(minutes=14)

    # Check if the current time is late
    if now > late_time_obj:
        cv2.putText(frame, f"Late Entry: {name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()

    c.execute("SELECT * FROM attendance WHERE name = ? AND date = ? AND subject = ?", (name, date, current_subject))
    existing_entry = c.fetchone()

    if existing_entry:
        print(f"Attendance for {name} on {date} for subject {current_subject} is already logged.")
        return frame

    # Late is 1 if we are late, if not then 0
    c.execute("INSERT INTO attendance (name, date, time, subject, late) VALUES (?, ?, ?, ?, ?)", 
              (name, date, time, current_subject, 1 if now > late_time_obj else 0))
    conn.commit()
    conn.close()

    print(f"Logged attendance for {name} on {date} at {time} for subject {current_subject}.")


    send_attendance_notification(name, date, time, current_subject)

    return frame



# COMPUTER VISION MAGIJA

def generate_frames():
    while True:
        # open camera capture and detect faces
        ret, frame = video_capture.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_image = frame[y:y+h, x:x+w]
            face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            inputs = processor(images=face_image_rgb, return_tensors="pt")
            with torch.no_grad():
                outputs = model.get_image_features(**inputs)
            face_embedding = outputs.cpu().numpy().flatten()
            face_embedding /= np.linalg.norm(face_embedding)  # Normalize the embedding

            if len(known_face_encodings) == 0:
                name = "Unknown"
                print("No known face encodings available.")
            else:
                # Find best match to detected face from known faces
                distances = np.linalg.norm(known_face_encodings - face_embedding, axis=1)
                min_distance_index = np.argmin(distances)
                name = "Unknown"
                if distances[min_distance_index] < 0.6:  # P
                    name = known_face_names[min_distance_index]
                    frame = log_attendance(name, frame)  # Log attendance with overlay if late

            # Bounding box + label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    api_key = os.getenv('WEATHER_API_KEY')
    location = "London" # Ili nešto drugo
    weather_condition = get_weather_forecast(api_key, location)
    
    if predict_absence_due_to_weather(weather_condition):
        message = "Bad weather predicted, late entries due to traffic problems are possible."
    else:
        message = "No significant weather issues expected. Students should come on time"
    
    return render_template('index.html', weather_condition=weather_condition, message=message)

@app.route('/attendance', methods=['GET'])
@login_required
def attendance():
    # Get filter parameters from the query string
    name_filter = request.args.get('name')
    subject_filter = request.args.get('subject')
    date_filter = request.args.get('date')
    weekday_filter = request.args.get('weekday')
    month_filter = request.args.get('month')
    year_filter = request.args.get('year')
    late_filter = request.args.get('late')

    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()

    # Build the SQL query dynamically based on the provided filters
    query = "SELECT rowid, subject, name, date, time, late FROM attendance WHERE 1=1"
    params = []

    if name_filter:
        query += " AND name = ?"
        params.append(name_filter)
    
    if subject_filter:
        query += " AND subject = ?"
        params.append(subject_filter)
    
    if date_filter:
        query += " AND date = ?"
        params.append(date_filter)
    
    if weekday_filter:
        query += " AND strftime('%w', date) = ?"
        params.append(weekday_filter)
    
    if month_filter:
        query += " AND strftime('%m', date) = ?"
        params.append(f"{int(month_filter):02d}")
    
    if year_filter:
        query += " AND strftime('%Y', date) = ?"
        params.append(year_filter)
    
    if late_filter:
        query += " AND late = ?"
        params.append(late_filter)

    query += " ORDER BY date, time"
    c.execute(query, params)
    
    records = c.fetchall()
    conn.close()

    # Group records by date and subject
    grouped_records = {}
    for rowid, subject, name, date, time, late in records:
        if date not in grouped_records:
            grouped_records[date] = {}
        if subject not in grouped_records[date]:
            grouped_records[date][subject] = []
        grouped_records[date][subject].append((rowid, name, time, late))
    
    return render_template('attendance.html', grouped_records=grouped_records)



@app.route('/download')
def download_attendance():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("SELECT subject, name, date, time FROM attendance ORDER BY subject, date, time")
    records = c.fetchall()
    conn.close()

    # Create a string buffer to write CSV data
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Initialize previous subject to keep track of subject changes
    previous_subject = None

    # Headeri
    writer.writerow(['Subject', 'Name', 'Date', 'Time'])

    #  CSV data grouped by subject
    for record in records:
        subject, name, date, time = record
        
        if subject != previous_subject:
            if previous_subject is not None:
                writer.writerow([])  # empty row between each subject
            writer.writerow([subject])  # subject name stavimo za header
            previous_subject = subject
        
        writer.writerow(['', name, date, time])
    
    # Seek to the beginning of the stream
    output.seek(0)
    
    return Response(output, mimetype="text/csv", headers={"Content-Disposition": "attachment;filename=attendance.csv"})


# isti vrag ko ovo gore, samo još šibnemo na mail
@app.route('/download_and_email')
def download_and_email_attendance():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("SELECT subject, name, date, time FROM attendance ORDER BY subject, date, time")
    records = c.fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)

    previous_subject = None

    writer.writerow(['Subject', 'Name', 'Date', 'Time'])

    for record in records:
        subject, name, date, time = record

        if subject != previous_subject:
            if previous_subject is not None:
                writer.writerow([])  
            writer.writerow([subject]) 
            previous_subject = subject

        writer.writerow(['', name, date, time])

    output.seek(0)

    csv_data = output.getvalue()

    # Assuming you have the user's email from the session or request
    user_email = request.args.get('email')  # Or get it from session

    return render_template('download.html', csv_data=csv_data, user_email=user_email)

# brisanje po id-ju
@app.route('/delete_attendance/<int:id>', methods=['POST'])
def delete_attendance(id):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("DELETE FROM attendance WHERE rowid = ?", (id,))
    conn.commit()
    conn.close()
    return redirect(url_for('attendance'))

# dohvat statistike...forši smislit još koju
@app.route('/statistics')
@login_required
def statistics():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()

    c.execute("SELECT name, subject, COUNT(*) FROM attendance GROUP BY name, subject")
    student_attendance = c.fetchall()

    c.execute("SELECT subject, COUNT(*) FROM attendance GROUP BY subject")
    subject_attendance = c.fetchall()

    conn.close()
    return render_template('statistics.html', student_attendance=student_attendance, subject_attendance=subject_attendance)

# initially debug ruta, ali san je pustija
@app.route('/students')
@login_required
def students():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("SELECT DISTINCT name FROM attendance ORDER BY name")
    students = c.fetchall()
    conn.close()

    return render_template('students.html', students=students)


# BITNO!!! NE NANKA POKUŠAVAT OPIRAT PLOTOVE AKO JOŠ NEMA ZABILJEŽENIH PRISUTNOSTI, JER ĆE SE ZBREJKAT
@app.route('/plot/student_attendance')
def plot_student_attendance():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()

    c.execute("SELECT name, COUNT(*) as count FROM attendance GROUP BY name")
    data = c.fetchall()
    conn.close()

    # Prepare data for plotting
    names, counts = zip(*data)

    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts, y=names, palette='viridis')
    plt.title('Attendance by Student')
    plt.xlabel('Number of Attendances')
    plt.ylabel('Student')

    # Save the plot to a BytesIO object and return it as a response
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return send_file(img, mimetype='image/png')

@app.route('/plot/subject_attendance')
def plot_subject_attendance():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()

    c.execute("SELECT subject, COUNT(*) as count FROM attendance GROUP BY subject")
    data = c.fetchall()
    conn.close()

    # Prepare data for plotting
    subjects, counts = zip(*data)

    # Create the plot
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=subjects, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
    plt.title('Attendance by Subject')

    # Save the plot to a BytesIO object and return it as a response
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return send_file(img, mimetype='image/png')


@app.route('/plot/monthly_attendance')
def plot_monthly_attendance():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()

    c.execute("SELECT date, COUNT(*) as count FROM attendance GROUP BY date")
    data = c.fetchall()
    conn.close()

    # Prepare data for plotting
    dates, counts = zip(*data)

    # Create the plot
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=dates, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
    plt.title('Monthly Attendance Distribution')

    # Save the plot to a BytesIO object and return it as a response
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return send_file(img, mimetype='image/png')

@app.route('/plots')
@login_required
def plots():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()

    # Query to count the number of attendance records
    c.execute('SELECT COUNT(*) FROM attendance')
    attendance_count = c.fetchone()[0]  # Get the count from the result
    conn.close()

    # Check if attendance_count is greater than 0
    if attendance_count > 0:
        return render_template('plot_router.html')
    else:
        flash("No attendance records found. Please add some attendance data before viewing the plots, because everything will break if you try to plot non-existing data <3", "error")
        return render_template('flash_redirect.html')  # Render a new template for displaying the message

        




# API CALL...AKO POKAŽE ODREĐENO VRIME, DAMO ALERT DA BI MOGLI KASNIT

def get_weather_forecast(api_key, location="your_city"):
    url = f"http://api.weatherapi.com/v1/forecast.json?key={api_key}&q={location}&days=1"
    response = requests.get(url)
    data = response.json()
    return data["forecast"]["forecastday"][0]["day"]["condition"]["text"]

def predict_absence_due_to_weather(weather_condition):
    bad_weather_keywords = ["rain", "storm", "snow", "fog", "hurricane"]
    for keyword in bad_weather_keywords:
        if keyword in weather_condition.lower():
            return True
    return False

@app.route('/predict_absence', methods=['GET'])
def predict_absence():
    api_key = "fe2e5f9339b2434db60124446241408"
    location = "London"
    weather_condition = get_weather_forecast(api_key, location)
    
    if predict_absence_due_to_weather(weather_condition):
        message = "Bad weather predicted, late entries due to traffic problems are possible."
    else:
        message = "No significant weather issues expected. Students should come on time"
    
    return jsonify({
        "weather_condition": weather_condition,
        "message": message
    })

# OVO JE KAO NEKI PROFESORSKI FORUM/CHAT ILI ČA JA ZNAN KAKO BI SE TO ZVALO
@app.route('/announcements', methods=['GET'])
@login_required
def announcements():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("SELECT * FROM announcements ORDER BY date_time ASC")
    announcements = c.fetchall()
    conn.close()
    return render_template('announcements.html', announcements=announcements)

@app.route('/post_announcement', methods=['POST'])
@login_required
def post_announcement():
    message = request.form['message']
    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    teacher_name = current_user.username

    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("INSERT INTO announcements (date_time, teacher_name, message) VALUES (?, ?, ?)",
              (date_time, teacher_name, message))
    conn.commit()
    conn.close()
    return redirect(url_for('announcements'))

@app.route('/delete_announcement/<int:id>', methods=['POST'])
@login_required
def delete_announcement(id):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("DELETE FROM announcements WHERE id = ?", (id,))
    conn.commit()
    conn.close()
    return redirect(url_for('announcements'))

@app.route('/edit_announcement/<int:id>', methods=['GET', 'POST'])
@login_required
def edit_announcement(id):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()

    if request.method == 'POST':
        message = request.form['message']
        date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        teacher_name = current_user.username

        c.execute("UPDATE announcements SET date_time = ?, message = ? WHERE id = ? AND teacher_name = ?",
                  (date_time, message, id, teacher_name))
        conn.commit()
        conn.close()
        return redirect(url_for('announcements'))
    
    c.execute("SELECT * FROM announcements WHERE id = ?", (id,))
    announcement = c.fetchone()
    conn.close()

    if announcement is None or announcement[2] != current_user.username:
        return redirect(url_for('announcements'))  # Redirect if announcement not found or not owned by user

    return render_template('edit_announcement.html', announcement=announcement)

# Ruta koja će najprije provjerit koliko predaanja je održano iz pojedinega predmeta
# Onda će provjerit za svakega studenta na koliko predavanja je bija
# To prikazat na način => veliki header=> naziv predmeta i ukupan broj predavanja, i ispod se izlistaju svi studenti koji su ikad bili na predmetu s brojen koliko puta su bili

@app.route('/report')
@login_required
def report():
    conn = sqlite3.connect('attendance.db')
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute('SELECT DISTINCT subject FROM attendance')
    subjects = cur.fetchall()

    report = []

    for subject in subjects:
        subject_name = subject['subject']

        cur.execute('SELECT COUNT(DISTINCT date) as total_lectures FROM attendance WHERE subject = ?', (subject_name,))
        total_lectures = cur.fetchone()['total_lectures']

        cur.execute('''SELECT name, COUNT(*) as attended_lectures, 
                       (COUNT(*) * 100.0 / ?) as attendance_percentage 
                       FROM attendance 
                       WHERE subject = ? 
                       GROUP BY name 
                       ORDER BY attendance_percentage DESC''', 
                   (total_lectures, subject_name))
        students_attendance = cur.fetchall()

        students_with_status = []
        for student in students_attendance:
            meets_requirement = student['attendance_percentage'] >= 50
            students_with_status.append({
                'name': student['name'],
                'attended_lectures': student['attended_lectures'],
                'attendance_percentage': student['attendance_percentage'],
                'meets_requirement': meets_requirement
            })

        cur.execute('''SELECT AVG(attendance_percentage) as avg_attendance 
                       FROM (SELECT COUNT(*) * 100.0 / ? as attendance_percentage 
                             FROM attendance 
                             WHERE subject = ? 
                             GROUP BY name)''', 
                   (total_lectures, subject_name))
        avg_attendance = cur.fetchone()['avg_attendance']

        report.append({
            'subject': subject_name,
            'total_lectures': total_lectures,
            'average_attendance': avg_attendance,
            'students': students_with_status
        })

    conn.close()

    return render_template('attendance_report.html', report=report)


# Ruta koja će dohvatit sva kašnjenja, s predmetima i entry timeon

@app.route('/late_analysis', methods=["GET", "POST"])
@login_required
def late_entries():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()

    # Fetch all late entries
    c.execute("SELECT * FROM attendance WHERE late = 1")
    late_entries = c.fetchall()
    conn.close()

    # Initialize counters
    hour_counter = Counter()
    weekday_counter = Counter()

    for entry in late_entries:
        time_in = entry[2]  # 'time' is the third column
        date = entry[1]     # 'date' is the second column

        # Convert time and date 
        time_obj = datetime.strptime(time_in, "%H:%M:%S")
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        # Count the hour
        hour_counter[time_obj.hour] += 1
        
        # Count the weekday (0=Monday, 6=Sunday)
        weekday_counter[date_obj.weekday()] += 1

    # Convert results to lists => rendering
    most_common_hour = hour_counter.most_common(1)[0] if hour_counter else None
    most_common_weekday = weekday_counter.most_common(1)[0] if weekday_counter else None

    # Show all hours from 00 to 23
    hours = list(range(24))
    hour_counts = [hour_counter.get(hour, 0) for hour in hours]  # Get count or 0 if not in the counter

    # Visualization
    if hour_counter:
        plt.bar(hours, hour_counts)
        plt.xticks(hours)  # Ensure all hours are labeled on the x-axis
        plt.xlabel('Hour of the Day')
        plt.ylabel('Number of Late Entries')
        plt.title('Late Entries by Hour')

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
    else:
        plot_url = None

    return render_template('late_entries.html', 
                           late_entries=late_entries, 
                           most_common_hour=most_common_hour, 
                           most_common_weekday=most_common_weekday,
                           plot_url=plot_url)


# WEBSCRAPING ROUTES => BEUTIFUL SOUP ZA STRANICE I PDFPLUMBER ZA PDF-OVE




def scrape_github_profile(url):
    try:
        # Send an HTTP request to the GitHub profile page
        response = requests.get(url)
        response.raise_for_status()

        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the name of the user
        name = soup.find('span', class_='p-name').text.strip()

        # Extract the bio (if available)
        bio = soup.find('div', class_='p-note user-profile-bio mb-3 js-user-profile-bio f4').text.strip() if soup.find('div', class_='p-note user-profile-bio mb-3 js-user-profile-bio f4') else 'No bio available'

        # Extract number of followers
        followers = soup.find('span', class_='text-bold').text.strip()

        return {
            'name': name,
            'bio': bio,
            'followers': followers,
        }
    except Exception as e:
        print(f"Error scraping the website: {e}")
        return None

@app.route('/scrape_github', methods=['GET'])
def github_profile():
    # GitHub profile URL to scrape
    url = 'https://github.com/AntonioLabinjan'
    
    if not url:
        return jsonify({"error": "Please provide a GitHub profile URL"}), 400
    
    # Scrape the profile
    profile_info = scrape_github_profile(url)
    
    if profile_info:
        # Render profile info using an HTML template
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>GitHub Profile</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: #111; /* Dark background */
                    color: #f9f9f9; /* Light text color */
                    margin: 0;
                    padding: 0;
                }}
                .container {{
                    max-width: 800px;
                    margin: 50px auto;
                    padding: 20px;
                    background-color: #222; /* Slightly lighter background for the container */
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
                    text-align: center;
                }}
                h1 {{
                    color: #ff6600; /* Orange color */
                }}
                p {{
                    font-size: 18px;
                    line-height: 1.6;
                }}
                .followers {{
                    font-weight: bold;
                    color: #ff6600; /* Orange color for followers text */
                    font-size: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{profile_info['name']}</h1>
                <p><strong>Bio:</strong> {profile_info['bio']}</p>
                <p class="followers">Followers: {profile_info['followers']}</p>
            </div>
        </body>
        </html>
        """
        return render_template_string(html_content), 200
    else:
        return jsonify({"error": "Failed to scrape the GitHub profile"}), 500



# Route to display the GitHub profile data in HTML template
def extract_pdf_text(pdf_url):
    response = requests.get(pdf_url, verify=False)
    if response.status_code == 200:
        with open("calendar.pdf", "wb") as f:
            f.write(response.content)
        
        # Extract text from the PDF using pdfplumber
        text_content = ""
        with pdfplumber.open("calendar.pdf") as pdf:
            for page in pdf.pages:
                text_content += page.extract_text() + "\n"
        return text_content
    else:
        return "Failed to retrieve PDF."

# filter only non-working days
def get_non_working_days(text):
    # Define keywords
    keywords = [
        "Blagdan", "Praznik", "nenastavni", "odmor", "Božić", "Nova Godina", 
        "Tijelovo", "Dan sjećanja", "Uskrs", "Svi sveti", "Sveta tri kralja", 
        "Dan državnosti", "Velike Gospe", "Dan domovinske zahvalnosti"
    ]
    
    # Search for dates with given keywords
    non_working_days = []
    for line in text.split("\n"):
        if any(keyword in line for keyword in keywords):
            non_working_days.append(line.strip())
    
    return "\n".join(non_working_days)

# Flask route to display the filtered non-working days
@app.route("/calendar")
def show_calendar():
    pdf_url = "https://www.unipu.hr/_download/repository/Sveu%C4%8Dili%C5%A1ni%20kalendar%20za%202024._2025..pdf"
    calendar_text = extract_pdf_text(pdf_url)
    non_working_days = get_non_working_days(calendar_text)
    
    # Split the non-working days into a list for the HTML unordered list
    non_working_days_list = non_working_days.split("\n")

    # Render the filtered non-working days as a list with bullet points
    html_content = f"""
    <html>
        <head>
            <title>Non-Working Days 2024/2025</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background-color: #f0f0f0;
                    color: #333;
                }}
                h1 {{
                    text-align: center;
                    color: #0056b3;
                }}
                ul {{
                    background-color: #ffffff;
                    padding: 20px;
                    border-radius: 8px;
                    border: 1px solid #ddd;
                    line-height: 1.6;
                    font-size: 14px;
                    list-style-type: disc;
                }}
                ul li {{
                    margin-bottom: 10px;
                }}
                .container {{
                    max-width: 900px;
                    margin: 0 auto;
                    padding: 20px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                    background-color: #ffffff;
                    border-radius: 8px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Non-Working Days 2024/2025</h1>
                <ul>
                    {''.join(f"<li>{day}</li>" for day in non_working_days_list if day.strip())}
                </ul>
            </div>
        </body>
    </html>
    """
    return render_template_string(html_content)



'''
Run the flask app on port 5144
'''
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5145, debug=True)
