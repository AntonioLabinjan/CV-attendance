import os
import cv2
import numpy as np
from flask import Flask, Response, render_template, send_file
import csv 
import io
from transformers import CLIPProcessor, CLIPModel
import torch
import sqlite3
from datetime import datetime

# Initialize CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# In-memory storage for known face embeddings and their labels
known_face_encodings = []
known_face_names = []

# Track attendance for the current session
logged_names = set()

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

app = Flask(__name__)

def create_db():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS attendance
                 (name TEXT, date TEXT, time TEXT, UNIQUE(name, date))''')
    conn.commit()
    conn.close()

create_db()


def log_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()

    # Check if attendance for the student on the current date already exists
    c.execute("SELECT * FROM attendance WHERE name = ? AND date = ?", (name, date))
    existing_entry = c.fetchone()
    
    if existing_entry:
        print(f"Attendance for {name} on {date} is already logged.")
        return  # If an entry exists, do nothing

    # If no existing entry, insert the new attendance record
    c.execute("INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)", (name, date, time))
    conn.commit()
    conn.close()

    print(f"Logged attendance for {name} on {date} at {time}.")


create_db()

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

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/attendance')
def attendance():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("SELECT name, date, time FROM attendance ORDER BY date, time")
    records = c.fetchall()
    conn.close()

    # Group records by date
    grouped_records = {}
    for name, date, time in records:
        if date not in grouped_records:
            grouped_records[date] = []
        grouped_records[date].append((name, time))
    
    return render_template('attendance.html', grouped_records=grouped_records)

@app.route('/download')
def download_attendance():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("SELECT name, date, time FROM attendance ORDER BY date, time")
    records = c.fetchall()
    conn.close()

    # Create a string buffer to write CSV data
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write CSV headers
    writer.writerow(['Name', 'Date', 'Time'])
    
    # Write CSV data
    for record in records:
        writer.writerow(record)
    
    # Seek to the beginning of the stream
    output.seek(0)
    
    return Response(output, mimetype="text/csv", headers={"Content-Disposition":"attachment;filename=attendance.csv"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5144)
