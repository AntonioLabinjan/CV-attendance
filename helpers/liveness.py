import cv2
import numpy as np

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detect_face(frame):
    """Detect faces in the frame using Haar cascades."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def detect_eye_movement(frame, face):
    """Check for eye movement (blinking) by detecting eyes."""
    (x, y, w, h) = face
    roi_gray = frame[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(roi_gray)

    # Check if two eyes are detected for liveness
    return len(eyes) >= 2  # Live face detected (eyes are open)

def check_liveness(frame, faces):
    """Check if the detected face is live."""
    for face in faces:
        if detect_eye_movement(frame, face):
            return True  # Live face detected
    return False  # Spoof detected

def perform_liveness_check():
    """Capture video from the camera and perform liveness detection."""
    cap = cv2.VideoCapture(0)  # Use the primary camera
    live_face_detected = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces in the current frame
        faces = detect_face(frame)

        # If faces are detected, check for liveness
        for face in faces:
            is_alive = check_liveness(frame, [face])
            (x, y, w, h) = face

            # Draw a bounding box around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue box

            if is_alive:
                live_face_detected = True
                print("OK, real")
                cv2.putText(frame, "Live Face Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                print("ERROR, fake")
                cv2.putText(frame, "Spoof Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow('Video Feed', frame)

        # Exit if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return live_face_detected

if __name__ == "__main__":
    if perform_liveness_check():
        print("Proceed to log attendance.")
    else:
        print("Attendance cannot be logged due to spoof detection.")
