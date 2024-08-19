import os
import cv2
import numpy as np
from flask import Flask, Response, render_template, send_file, request, redirect, url_for
import csv
import io
from transformers import CLIPProcessor, CLIPModel
import torch
import sqlite3
from datetime import datetime, timedelta

# Initialize CLIP model and processor
'''
The CLIP (Contrastive Language–Image Pre-training) model and processor are initialized here. The CLIPModel and CLIPProcessor are both part of the transformers library, which allows the model to process images and generate embeddings.
CLIPModel: This is the pre-trained model used to generate image features (embeddings) from input images.
CLIPProcessor: This helps prepare the images so they can be processed by the CLIP model, ensuring the images are in the correct format.
These components are crucial for facial recognition as they generate unique embeddings for faces, which can later be compared to identify individuals.
'''
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# In-memory storage for known face embeddings and their labels
'''
These lists store data about faces that the system recognizes.
known_face_encodings: A list to store the embeddings (numerical representations) of known faces. Each face's embedding is unique and is used for comparison against detected faces.
known_face_names: A list that stores the names corresponding to each face encoding. These names are matched with the embeddings in known_face_encodings based on index.

When a face is detected, its embedding is generated using the CLIP model, and this embedding is compared against the embeddings in known_face_encodings. If a match is found (based on similarity), the corresponding name from known_face_names is used to identify the person.
'''
known_face_encodings = []
known_face_names = []

# Track attendance for the current session
'''
This set keeps track of which students have already been marked as present in the current session to prevent duplicate entries.
Once a student's attendance is logged, their name is added to this set. Before logging attendance for a detected face, the system checks this set to ensure the student's attendance isn't logged multiple times in a single session.
'''

logged_names = set()

# Function to add a known face
'''
The add_known_face function processes an image to generate a unique embedding (numerical representation) for a face using the CLIP model and stores it for 
future recognition. It begins by loading and converting the image to RGB format, then processes it with a CLIP processor to prepare it as a tensor for the model. 
The model generates an embedding, which is normalized and added to a list of known face encodings, with the corresponding name stored separately. 
This enables the system to recognize the face in future sessions, and the function confirms successful addition with a printed message
'''

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
'''
The load_known_faces function iterates through a directory named known_faces, where each subdirectory represents a student, 
and each file within contains an image of that student's face. For each image found, it calls the add_known_face function to generate and store its face embedding.
 If an image fails to load, it catches and prints the error. After processing all the images, it outputs the total number of loaded face encodings and their 
 corresponding names. This function is called at startup to preload all known faces into memory, ensuring they are available for recognition during the session.
'''
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

app = Flask(__name__)

app.secret_key = 'DO_NOT_VISIT_GRMIALDA'

'''
The create_db function initializes an SQLite database named attendance.db. It first establishes a connection to the database and creates a cursor object 
for executing SQL commands. The function drops any existing attendance table, ensuring that the table is reset every time the function is run. 
It then creates a new attendance table with columns for storing a student's name, the date and time of attendance, the subject, and a boolean flag indicating if the student was late.
The UNIQUE constraint prevents duplicate attendance records for the same student, date, and subject. After executing the SQL commands, the function commits the changes and closes the connection. 
This function is called to set up the database at the start of the application.
 Additionally, variables for the current subject, attendance date, start time, and end time are initialized to None for future use.
'''
def create_db():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute('''DROP TABLE IF EXISTS attendance''')
    c.execute('''CREATE TABLE IF NOT EXISTS attendance
                 (name TEXT, date TEXT, time TEXT, subject TEXT, late BOOLEAN DEFAULT 0, UNIQUE(name, date, subject))''')
    conn.commit()
    conn.close()

create_db()

# Initialize current subject details
'''
Variables for details are initialized at None, so we can set them later
'''
current_subject = None
attendance_date = None
start_time = None
end_time = None

'''
The /set_subject route is responsible for setting the current logging subject and the associated attendance parameters. 
It handles both GET and POST requests. When accessed via GET, it renders a form (set_subject.html) for the user to input details.
 When a POST request is submitted, the route extracts the subject, date, start time, and end time from the form data and updates the global variables current_subject, attendance_date, start_time, and end_time with these values. 
 After updating these variables, it renders a success template (set_subject_success.html), displaying the new subject and time settings. 
This functionality allows the application to dynamically configure attendance logging details based on user input.
'''
# route to set current logging subject
@app.route('/set_subject', methods=['GET', 'POST'])
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

'''
The is_within_time_interval function checks whether the current date and time fall within the specified range for logging attendance. 
It retrieves the current date and time using datetime.now(), then formats them to match the format used in the attendance_date, start_time, and end_time variables.
 The function returns True if the current date matches the attendance_date and the current time is within the range defined by start_time and end_time; otherwise, it returns False.
 This ensures that attendance can only be recorded if it is within the predefined time window for the specified subject.
'''
# check if current date&time is appropriate to log certain subject
def is_within_time_interval():
    now = datetime.now()
    current_time = now.strftime("%H:%M")
    current_date = now.strftime("%Y-%m-%d")
    return (current_date == attendance_date and 
            start_time <= current_time <= end_time)


'''
The `add_student` function handles both displaying the form for adding a new student and processing the form submission. 
When a POST request is made, it extracts the student's name and images from the form data. It then creates a subfolder in the `known_faces` directory named after the student to store their images. 
Each uploaded image is saved to this subfolder and processed to add the student's face to the known face database using the `add_known_face` function.
 After processing, it renders a success page to confirm the addition. For GET requests, it simply renders the form for adding a student.
 The `add_student_success` route serves the success page template to display confirmation once a student has been added.
'''
# add a new student to database and add his images to folder for clip recognition
@app.route('/add_student', methods=['GET', 'POST'])
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

# Add student success route (render a success template when student is added)
@app.route('/add_student_success')
def add_student_success():
    return render_template('add_student_success.html')

'''
The log_attendance function records the attendance of a recognized person if the current subject is set and the time falls within the defined interval.
 It first checks if the current_subject is defined and if the current time is within the valid range using the is_within_time_interval function. 
 If not, it logs a message and returns the frame without making any changes.
If the conditions are met, it calculates whether the person is late based on a 14-minute threshold after the start time.
It then checks the database to ensure that attendance for this person on this date and subject has not already been recorded to avoid duplicates. 
If the record does not exist, it inserts a new attendance entry, marking it as late if the current time exceeds the threshold. 
Finally, it prints a confirmation message and returns the frame, possibly with an overlay indicating a late entry.
'''

# Function to log attendance for recognized person
def log_attendance(name, frame):
    global current_subject, attendance_date, start_time, end_time
    if current_subject is None or not is_within_time_interval():
        print("Subject is not set or current time is outside of allowed interval. Attendance not logged.")
        return frame

# We log attendance on current date at current time
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    # Create datetime object for start time
    start_time_obj = datetime.strptime(f"{attendance_date} {start_time}", "%Y-%m-%d %H:%M")

    # Calculate the late threshold time
    late_time_obj = start_time_obj + timedelta(minutes=14)

    # Check if the current time is late
    if now > late_time_obj:
        cv2.putText(frame, f"Late Entry: {name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()

    # Insert attendance in database for recognized person on chosen subject
    c.execute("SELECT * FROM attendance WHERE name = ? AND date = ? AND subject = ?", (name, date, current_subject))
    existing_entry = c.fetchone()

    # Disable duplicate inputs (one person can be logged only once per day on a certain subject)
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

# Function for generating video frames using camera capture and face recognition using clip model
'''
The generate_frames function captures video frames from a camera, performs face detection, and applies face recognition using the CLIP model. 
It continuously reads frames from the camera, converts them to grayscale for face detection using Haar cascades, and identifies faces in each frame. 
For each detected face, it extracts the face image, processes it with the CLIP model to obtain an embedding, and normalizes the embedding. 
It then compares this embedding with known face embeddings to find the best match. If a known face is recognized within a certain threshold, it logs the attendance and potentially marks it as late.
 Each frame is annotated with a rectangle around the face and a label with the recognized name, and is then encoded as a JPEG image to be streamed as part of a video feed.
'''
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
            # take face images for model inputs
            inputs = processor(images=face_image_rgb, return_tensors="pt")
            with torch.no_grad():
                outputs = model.get_image_features(**inputs)
            # reshape of face embeddings so they can be processed by clip mofrl    
            face_embedding = outputs.cpu().numpy().flatten()
            face_embedding /= np.linalg.norm(face_embedding)  # Normalize the embedding

            # if there are no faces in database
            if len(known_face_encodings) == 0:
                name = "Unknown"
                print("No known face encodings available.")
            else:
                # If there are faces in db, we find the best match for the detected face
                distances = np.linalg.norm(known_face_encodings - face_embedding, axis=1)
                min_distance_index = np.argmin(distances)
                name = "Unknown"
                if distances[min_distance_index] < 0.6:  # Adjusted threshold
                    name = known_face_names[min_distance_index]
                    frame = log_attendance(name, frame)  # Log attendance with overlay if late

            # Draw a rectangle around the face and label it with recognized person name
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# route for video capture and face recognition
'''
this loads generated frames from videocamera capture with bounding boxes&labels
'''
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# main route which renders main html template
'''
load main route with homepage
'''
@app.route('/')
def index():
    return render_template('index.html')

# route to fetch all attendance records from database
'''
The attendance route retrieves all attendance records from the database, organizes them by date and subject, and then renders them in a structured format.
 It connects to the SQLite database and executes a query to fetch records, including the subject, student name, date, time, and whether the student was late. 
 The function then groups these records first by date and then by subject within each date to facilitate easier viewing and analysis.
 The organized data is passed to the attendance.html template for display, providing a clear overview of attendance records.
'''
@app.route('/attendance')
def attendance():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    # each record contains of subject, student's name, current date&time and bool variable late which logs if student is late
    c.execute("SELECT rowid, subject, name, date, time, late FROM attendance ORDER BY date, time")
    records = c.fetchall()
    conn.close()

    # Group records by date and subject for clearer output
    grouped_records = {}
    for rowid, subject, name, date, time, late in records:
        if date not in grouped_records:
            grouped_records[date] = {}
        if subject not in grouped_records[date]:
            grouped_records[date][subject] = []
        grouped_records[date][subject].append((rowid, name, time, late))
    
    return render_template('attendance.html', grouped_records=grouped_records)

# route to fetch records from database and export them in csv format
'''
The `download_attendance` route generates and returns a CSV file containing all attendance records from the database. 
It connects to the SQLite database, retrieves attendance data (including subject, student name, date, and time), and organizes it by subject.
The function uses an in-memory string buffer (`io.StringIO`) to create and format the CSV file.
It writes headers and data into the buffer, grouping records by subject with appropriate section headers and separating subjects with blank rows.
Finally, the CSV data is returned as a downloadable file with the MIME type set to "text/csv" and a filename of `attendance.csv`, allowing users to download the attendance records.
'''
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

# route to delete attendance from db by row id
'''
The delete_attendance route handles the deletion of a specific attendance record from the database.
It accepts a POST request with a rowid parameter indicating the record to be deleted. The function connects to the SQLite database,
executes a SQL DELETE command to remove the record corresponding to the provided rowid, and commits the changes.
After successfully deleting the record, the function closes the database connection and redirects the user to the /attendance page,
which updates the view to reflect the removal of the record.
'''
@app.route('/delete_attendance/<int:id>', methods=['POST'])
def delete_attendance(id):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("DELETE FROM attendance WHERE rowid = ?", (id,))
    conn.commit()
    conn.close()
    return redirect(url_for('attendance'))

# route for statistics. Shows number of attendances for each student and for each subject
'''
The statistics route retrieves and displays attendance statistics.
It connects to the SQLite database and executes two SQL queries. 
The first query counts the number of attendance records for each student per subject and groups the results by student name and subject. 
The second query counts the total number of attendance records for each subject. Both sets of results are fetched and stored in variables. 
After closing the database connection, the function renders the statistics.html template, passing the collected data (student_attendance and subject_attendance)
to be displayed. This allows users to view detailed statistics on attendance by student and subject.
'''
@app.route('/statistics')
def statistics():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()

    c.execute("SELECT name, subject, COUNT(*) FROM attendance GROUP BY name, subject")
    student_attendance = c.fetchall()

    c.execute("SELECT subject, COUNT(*) FROM attendance GROUP BY subject")
    subject_attendance = c.fetchall()

    conn.close()
    return render_template('statistics.html', student_attendance=student_attendance, subject_attendance=subject_attendance)

# route which fetches all students and displays their names
'''
The students route fetches and displays the list of unique student names from the database.
It connects to the SQLite database and executes a SQL query to select distinct student names from the attendance table, ordered alphabetically.
After retrieving the list of student names, the function closes the database connection and renders the students.html template, passing the fetched names as a variable.
This allows users to view a sorted list of all students who have been recorded in the attendance system.
'''
@app.route('/students')
def students():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("SELECT DISTINCT name FROM attendance ORDER BY name")
    students = c.fetchall()
    conn.close()

    return render_template('students.html', students=students)

'''
Run the flask app on port 5144
'''
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5144)
