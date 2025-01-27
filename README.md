Docker run command: docker run -p 5145:5145 --name majkabojza my-attendance-app
CHECKPOINT.ipynb is important
# important colab stuff: 
- https://colab.research.google.com/drive/1nrCUORercQ2naDgDDbWI6EDxUA9TBVRa?authuser=2#scrollTo=u-VfIT50OcYj
- https://drive.google.com/drive/u/2/folders/1A4OaZsdmRrFKBuv-KGyu9Pdko9iPuTw5
- https://colab.research.google.com/drive/1ZKVMC7qs3HBf3a2hHpYBdOiWyWDQUK8O#scrollTo=_O9W2yRrTOXn
#### Čudan pokušaj tuninga clip-a: https://colab.research.google.com/drive/19LnwteQm3fQDx77hlOybQwT5xq5t6Yk1?usp=sharing#scrollTo=CmTFz3RObaWr
### NAPOMENA: lokalni folder di sve ovo delan se zove WORKING_CV_ATTENDANCE (da ne zabin hehe)
# Face Recognition Attendance System

# sendgrid recovery code: XYWZZG4535Y3ELMJJL6TRRVM
Test credentials:
Email: alabinjan6@gmail.com
Username: Antonio
Password: 4uGnsUh9!!!


![Face Recognition Attendance System](https://img.shields.io/badge/Face_Recognition_Attendance_System-v1.0-brightgreen)

A simple and effective Face Recognition Attendance System using CLIP model for face embeddings and OpenCV for real-time face detection. This project captures and logs attendance in real-time based on recognized faces.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Folder Structure](#folder-structure)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Project Overview

This Face Recognition Attendance System enables automated attendance tracking using face recognition technology. It captures live video from a webcam, detects faces, recognizes them, and logs attendance in an SQLite database. The system features a simple web interface for monitoring and viewing attendance records.

### Features

- **Real-Time Face Detection**: Uses Haar Cascade Classifier to detect faces in real-time from webcam feed.
- **Face Recognition**: Utilizes the CLIP model from Hugging Face to recognize and match faces.
- **Attendance Logging**: Logs each student's attendance in an SQLite database, ensuring that each student is only logged once per session.
- **Web Interface**: Provides a web-based interface to view the live camera feed and attendance records.

## Installation

### Prerequisites

1. **Python**: Ensure you have Python 3.8+ installed.
2. **Pip**: Ensure you have pip for installing Python packages.

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/face-recognition-attendance.git
   cd face-recognition-attendance
   
2. **Create and activate VEnv** (optional)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. **Install dependecies**
pip install -r requirements.txt

5. **Set up database**
The database will be automatically created on the first run. Ensure attendance.db is writable by the application.

6. **Prepare known faces**
Place student images in the known_faces directory, organized by student folders.


Video demo: https://youtu.be/hQDcAjGRHMQ => OFFICIAL VIDEO ZA PROJEKT
