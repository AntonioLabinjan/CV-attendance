import os
import cv2
import numpy as np
from flask import Flask, Response, render_template, send_file, request, redirect, url_for, session
import csv 
import io
from transformers import CLIPProcessor, CLIPModel
import torch
import sqlite3
from datetime import datetime
from functools import wraps

# Initialize CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# In-memory storage for known face embeddings and their labels
known_face_encodings = []
known_face_names = []

# Track attendance for the current session
logged_names = set()

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for session management

def create_db():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    
    # Drop tables for demonstration purposes; you should use ALTER or other methods in production
    c.execute('''DROP TABLE IF EXISTS attendance''')
    c.execute('''DROP TABLE IF EXISTS users''')
    c.execute('''DROP TABLE IF EXISTS student_subjects''')  # New table for student-subject relations
    c.execute('''DROP TABLE IF EXISTS students''')

    c.execute('''CREATE TABLE IF NOT EXISTS students
                 (name TEXT PRIMARY KEY)''')

    c.execute('''CREATE TABLE IF NOT EXISTS attendance
                 (name TEXT, date TEXT, time TEXT, subject TEXT, UNIQUE(name, date, subject))''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, password TEXT, role TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS student_subjects
                 (student_name TEXT, subject TEXT)''')  # Table to track which subjects a student is registered for
    
    # Create a default admin user
    c.execute("INSERT INTO users (username, password, role) VALUES ('admin', 'adminpass', 'professor')")
    
    conn.commit()
    conn.close()

create_db()


# Helper function to check if the user is logged in
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Helper function to check if the user is a professor
def professor_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'role' not in session or session['role'] != 'professor':
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('attendance.db')
        c = conn.cursor()
        c.execute("SELECT username, role FROM users WHERE username = ? AND password = ?", (username, password))
        user = c.fetchone()
        conn.close()

        if user:
            session['username'] = user[0]
            session['role'] = user[1]
            
            # Redirect based on user role
            if user[1] == 'professor':
                return redirect(url_for('set_subject'))  # Redirect to the admin panel
            else:
                return redirect(url_for('index'))  # Redirect to homepage for students

        else:
            return "Invalid credentials"

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/')
@login_required
def index():
    return render_template('index.html', role=session['role'])

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

@app.route('/set_subject', methods=['GET', 'POST'])
@login_required
@professor_required  # Only professors can set subjects
def set_subject():
    global current_subject
    if request.method == 'POST':
        current_subject = request.form['subject']
        return render_template('set_subject_success.html', subject=current_subject)
    return render_template('set_subject.html')

@app.route('/add_student', methods=['GET', 'POST'])
@login_required
@professor_required  # Only professors can add students
def add_student():
    if request.method == 'POST':
        name = request.form['name']
        images = request.files.getlist('images')
        subjects = request.form.getlist('subjects')  # Get list of subjects

        # Create a new subfolder in the known_faces directory
        student_dir = os.path.join('known_faces', name)
        os.makedirs(student_dir, exist_ok=True)

        for image in images:
            # Save each uploaded image in the new student's subfolder
            image_path = os.path.join(student_dir, image.filename)
            image.save(image_path)
            add_known_face(image_path, name)

        # Register student and their subjects in the database
        conn = sqlite3.connect('attendance.db')
        c = conn.cursor()
        
        # Add student to students table
        c.execute("INSERT OR IGNORE INTO students (name) VALUES (?)", (name,))
        
        # Add subjects to student_subjects table
        c.executemany("INSERT INTO student_subjects (student_name, subject) VALUES (?, ?)",
                      [(name, subject) for subject in subjects])
        
        conn.commit()
        conn.close()

        return render_template('add_student_success.html', name=name)

    # Fetch available subjects (assuming you have a list or a way to get these)
    available_subjects = ["Programiranje", "Baze podataka II", "Web aplikacije"]  # Replace with actual subjects

    return render_template('add_student.html', available_subjects=available_subjects)




@app.route('/video_feed')
@login_required # we don't need professor pass for this
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/attendance')
@login_required
@professor_required  # Only professors can view attendance
def attendance():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("SELECT rowid, subject, name, date, time FROM attendance ORDER BY date, time")
    records = c.fetchall()
    conn.close()

    # Group records by date and subject
    grouped_records = {}
    for rowid, subject, name, date, time in records:
        if date not in grouped_records:
            grouped_records[date] = {}
        if subject not in grouped_records[date]:
            grouped_records[date][subject] = []
        grouped_records[date][subject].append((rowid, name, time))
    
    return render_template('attendance.html', grouped_records=grouped_records)

@app.route('/download')
@login_required
@professor_required  # Only professors can download attendance
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
@login_required
@professor_required  # Only professors can delete attendance records
def delete_attendance(id):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("DELETE FROM attendance WHERE rowid = ?", (id,))
    conn.commit()
    conn.close()
    return redirect(url_for('attendance'))

@app.route('/statistics')
@login_required
@professor_required  # Only professors can view statistics
def statistics():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()

    c.execute("SELECT name, subject, COUNT(*) FROM attendance GROUP BY name, subject")
    student_attendance = c.fetchall()

    c.execute("SELECT subject, COUNT(*) FROM attendance GROUP BY subject")
    subject_attendance = c.fetchall()

    conn.close()
    return render_template('statistics.html', student_attendance=student_attendance, subject_attendance=subject_attendance)

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
                    log_attendance(name)  # Log attendance

            # Draw a rectangle around the face and label it
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def log_attendance(name):
    global current_subject
    if current_subject is None:
        print("No subject set. Attendance not logged.")
        return

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()

    # Check if the student is registered for the current subject
    c.execute("SELECT * FROM student_subjects WHERE student_name = ? AND subject = ?", (name, current_subject))
    registration = c.fetchone()

    if not registration:
        print(f"{name} is not registered for the subject {current_subject}. Attendance not logged.")
        return  # Exit if the student is not registered for the subject

    # Check if attendance for the student on the current date and subject already exists
    c.execute("SELECT * FROM attendance WHERE name = ? AND date = ? AND subject = ?", (name, date, current_subject))
    existing_entry = c.fetchone()
    
    if existing_entry:
        print(f"Attendance for {name} on {date} for subject {current_subject} is already logged.")
        return  # If an entry exists, do nothing

    # If no existing entry, insert the new attendance record
    c.execute("INSERT INTO attendance (name, date, time, subject) VALUES (?, ?, ?, ?)", (name, date, time, current_subject))
    conn.commit()
    conn.close()

    print(f"Logged attendance for {name} on {date} at {time} for subject {current_subject}.")



@app.route('/list_students')
#@login_required
#@professor_required  # Ensure only professors can access this
def list_students():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("SELECT s.name, ss.subject FROM students s JOIN student_subjects ss ON s.name = ss.student_name ORDER BY s.name")
    students = c.fetchall()
    conn.close()

    return render_template('list_students.html', students=students)


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



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5144)
