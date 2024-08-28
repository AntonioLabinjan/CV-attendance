import os
import cv2
import numpy as np
from flask import Flask, Response, render_template, send_file, request, redirect, url_for, jsonify
import csv
import io
from transformers import CLIPProcessor, CLIPModel
import torch
import sqlite3
import datetime
from datetime import datetime, timedelta
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import requests

# Initialize CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# In-memory storage for known face embeddings and their labels
known_face_encodings = []
known_face_names = []

# Track attendance for the current session
logged_names = set()

app = Flask(__name__)

# ovo bi bilo dobro spremit u .env fajl, ali to ću ben delat
app.secret_key = 'DO_NOT_VISIT_GRMIALDA'


# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User model
class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

# In-memory user storage (can be replaced with a database)
users = {}

@login_manager.user_loader
def load_user(user_id):
    return users.get(int(user_id))

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

    # Debugging output
    print(f"Loaded {len(known_face_encodings)} known face encodings")
    print(f"Known face names: {known_face_names}")

# Load known faces at startup
load_known_faces()

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

# Face detection using Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def create_db():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute('''DROP TABLE IF EXISTS attendance''')
    c.execute('''CREATE TABLE IF NOT EXISTS attendance
                 (name TEXT, date TEXT, time TEXT, subject TEXT, late BOOLEAN DEFAULT 0, UNIQUE(name, date, subject))''')
    
    c.execute('''DROP TABLE IF EXISTS announcements''')
    c.execute('''CREATE TABLE IF NOT EXISTS announcements
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  date_time TEXT, 
                  teacher_name TEXT, 
                  message TEXT)''')
    conn.commit()
    conn.close()

create_db()

# Initialize current subject details
current_subject = None
attendance_date = None
start_time = None
end_time = None

from werkzeug.security import generate_password_hash, check_password_hash

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        user_id = len(users) + 1
        new_user = User(user_id, username, hashed_password)
        users[user_id] = new_user

        return redirect(url_for('login'))
    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = next((u for u in users.values() if u.username == username), None)
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))

        return render_template('login.html', error="Invalid username or password")

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
    now = datetime.datetime.now()
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
        
        # Create a new subfolder in the known_faces directory
        student_dir = os.path.join('known_faces', name)
        os.makedirs(student_dir, exist_ok=True)

        for image in images:
            # Save each uploaded image in the new student's subfolder
            image_path = os.path.join(student_dir, image.filename)
            image.save(image_path)
            add_known_face(image_path, name)
        
        return render_template('add_student_success.html', name=name)

    return render_template('add_student.html')

# Add student success route
@app.route('/add_student_success')
def add_student_success():
    return render_template('add_student_success.html')

def log_attendance(name, frame):
    global current_subject, attendance_date, start_time, end_time
    if current_subject is None or not is_within_time_interval():
        print("Subject is not set or current time is outside of allowed interval. Attendance not logged.")
        return frame

    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    # Create datetime object for start time
    start_time_obj = datetime.datetime.strptime(f"{attendance_date} {start_time}", "%Y-%m-%d %H:%M")

    # Calculate the late threshold time
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

    # Insert the new attendance record
    c.execute("INSERT INTO attendance (name, date, time, subject, late) VALUES (?, ?, ?, ?, ?)", 
              (name, date, time, current_subject, 1 if now > late_time_obj else 0))
    conn.commit()
    conn.close()

    print(f"Logged attendance for {name} on {date} at {time} for subject {current_subject}.")

    return frame

def generate_frames():
    while True:
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
                # Find the best match for the detected face
                distances = np.linalg.norm(known_face_encodings - face_embedding, axis=1)
                min_distance_index = np.argmin(distances)
                name = "Unknown"
                if distances[min_distance_index] < 0.6:  # Adjusted threshold
                    name = known_face_names[min_distance_index]
                    frame = log_attendance(name, frame)  # Log attendance with overlay if late

            # Draw a rectangle around the face and label it
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
    api_key = "fe2e5f9339b2434db60124446241408"
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

    # Write CSV headers
    writer.writerow(['Subject', 'Name', 'Date', 'Time'])

    # Write CSV data grouped by subject
    for record in records:
        subject, name, date, time = record
        
        if subject != previous_subject:
            if previous_subject is not None:
                writer.writerow([])  # Add an empty row for separation between subjects
            writer.writerow([subject])  # Write the subject name as a new section header
            previous_subject = subject
        
        writer.writerow(['', name, date, time])
    
    # Seek to the beginning of the stream
    output.seek(0)
    
    return Response(output, mimetype="text/csv", headers={"Content-Disposition": "attachment;filename=attendance.csv"})

@app.route('/delete_attendance/<int:id>', methods=['POST'])
def delete_attendance(id):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("DELETE FROM attendance WHERE rowid = ?", (id,))
    conn.commit()
    conn.close()
    return redirect(url_for('attendance'))

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

@app.route('/students')
@login_required
def students():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("SELECT DISTINCT name FROM attendance ORDER BY name")
    students = c.fetchall()
    conn.close()

    return render_template('students.html', students=students)

import matplotlib.pyplot as plt
import seaborn as sns
from flask import send_file
import io


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

    # Adjust this query based on your actual schema. Assuming you want to group by date.
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
    return render_template('plot_router.html')



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
    
    # Return both the weather condition and the prediction message
    return jsonify({
        "weather_condition": weather_condition,
        "message": message
    })

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
    # Establish connection to the database
    conn = sqlite3.connect('attendance.db')
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Fetch all distinct subjects
    cur.execute('SELECT DISTINCT subject FROM attendance')
    subjects = cur.fetchall()

    report = []

    for subject in subjects:
        subject_name = subject['subject']

        # Count the total number of lectures for the subject
        cur.execute('SELECT COUNT(DISTINCT date) as total_lectures FROM attendance WHERE subject = ?', (subject_name,))
        total_lectures = cur.fetchone()['total_lectures']

        # Count the number of lectures attended by each student for this subject
        cur.execute('''SELECT name, COUNT(*) as attended_lectures 
                       FROM attendance 
                       WHERE subject = ? 
                       GROUP BY name''', 
                    (subject_name,))
        students_attendance = cur.fetchall()

        # Add percentage calculation and requirement check for each student
        students_with_status = []
        for student in students_attendance:
            attendance_percentage = (student['attended_lectures'] / total_lectures) * 100
            meets_requirement = attendance_percentage >= 50
            students_with_status.append({
                'name': student['name'],
                'attended_lectures': student['attended_lectures'],
                'attendance_percentage': attendance_percentage,
                'meets_requirement': meets_requirement
            })

        # Prepare the report for this subject
        report.append({
            'subject': subject_name,
            'total_lectures': total_lectures,
            'students': students_with_status
        })

    # Close the connection
    conn.close()

    # Render the report using the HTML template
    return render_template('attendance_report.html', report=report)


# Tu ubacit još malo neke more advanced analize :) i obavezno vizalizaciju

import datetime
from collections import Counter

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
        time_in = entry[2]  # Assuming the 'time' is the third column
        date = entry[1]     # Assuming the 'date' is the second column

        # Print time_in for debugging
        print(f"time_in: {time_in}")  # Debugging line

        # Convert time and date to appropriate formats
        time_obj = datetime.datetime.strptime(time_in, "%H:%M:%S")  # Updated format
        date_obj = datetime.datetime.strptime(date, "%Y-%m-%d")
        
        # Count the hour
        hour_counter[time_obj.hour] += 1
        
        # Count the weekday (0=Monday, 6=Sunday)
        weekday_counter[date_obj.weekday()] += 1

    # Convert results to lists for template rendering
    most_common_hour = hour_counter.most_common(1)[0] if hour_counter else None
    most_common_weekday = weekday_counter.most_common(1)[0] if weekday_counter else None

    return render_template('late_entries.html', 
                           late_entries=late_entries, 
                           most_common_hour=most_common_hour, 
                           most_common_weekday=most_common_weekday)


'''
Run the flask app on port 5144
'''
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5144, debug=True)
