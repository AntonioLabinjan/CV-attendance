# ČA TREBA NAPRAVIT?
# POSTAVIT THRESHOLD KOJI JE POTREBAN DA UOPĆE RAZMATRAMO TENSOR KAO VALID MATCH => UBRAZAMO PROCES JER NE VOTAMO IZMEĐU SVIH NEGO IZMEĐU ODREĐENIH TOP K (tj. uzimamo top k od top k) => ALI DOHVATIMO IH VIŠE (prvi K je veći, onda od njega uzmimamo manji K)
# DELA. VIŠE NE TRAŽIMO TOP K OD SVIH, NEGO TOP K OD ODREĐENEGA SUBSETA
import tkinter as tk
from tkinter import simpledialog
import cv2
import numpy as np
import torch
import faiss
from transformers import CLIPProcessor, CLIPModel
from collections import Counter
import os

# Resolve OpenMP runtime error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

known_face_encodings = []
known_face_names = []

def add_known_face(image_path, name):
    print(f"Processing image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} not found.")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inputs = processor(images=image_rgb, return_tensors="pt")
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    embedding = outputs.cpu().numpy().flatten()
    known_face_encodings.append(embedding / np.linalg.norm(embedding))
    known_face_names.append(name)
    print(f"Added face for {name} from {image_path}")

def load_known_faces():
    print("Loading known faces...")
    for person_name in os.listdir("../known_faces"):
        person_dir = os.path.join("../known_faces", person_name)
        if os.path.isdir(person_dir):
            print(f"Processing directory: {person_dir}")
            for filename in os.listdir(person_dir):
                image_path = os.path.join(person_dir, filename)
                try:
                    add_known_face(image_path, person_name)
                except ValueError as e:
                    print(f"Error: {e}")
    print(f"Total faces loaded: {len(known_face_names)}")

def build_index(known_face_encodings):
    known_face_encodings = np.array(known_face_encodings)
    dimension = known_face_encodings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(known_face_encodings)
    return faiss_index

def classify_face(face_embedding, faiss_index, known_face_names, k1, k2, threshold):
    distances, indices = faiss_index.search(face_embedding[np.newaxis, :], k1)
    valid_indices = [i for i, dist in zip(indices[0], distances[0]) if dist < threshold]
    filtered_indices = valid_indices[:k2] if len(valid_indices) > k2 else valid_indices
    top_matches = [known_face_names[idx] for idx in filtered_indices]
    class_counts = Counter(top_matches)
    if not class_counts:
        return "Unknown", class_counts
    majority_class = class_counts.most_common(1)[0][0]
    return majority_class, class_counts

class FaceRecognitionApp:
    def __init__(self):
        self.k1 = 50
        self.k2 = 20
        self.threshold = 0.5
        self.faiss_index = None
        self.load_faces_on_startup()
        self.setup_ui()

    def load_faces_on_startup(self):
        load_known_faces()
        global known_face_encodings
        self.faiss_index = build_index(known_face_encodings)
        print(f"Faces Loaded: {len(known_face_names)} known faces")

    def setup_ui(self):
        self.root = tk.Tk()
        self.root.title("Face Recognition App")
        tk.Button(self.root, text="Set k1", command=self.set_k1).pack()
        tk.Button(self.root, text="Set k2", command=self.set_k2).pack()
        tk.Button(self.root, text="Set Threshold", command=self.set_threshold).pack()
        tk.Button(self.root, text="Start Camera", command=self.start_camera).pack()
        tk.Button(self.root, text="Quit", command=self.quit_app).pack()
        tk.Label(self.root, text=f"Faces Loaded: {len(known_face_names)}").pack()
        self.root.mainloop()

    def set_k1(self):
        self.k1 = simpledialog.askinteger("Set k1", "Enter k1:", initialvalue=self.k1)

    def set_k2(self):
        self.k2 = simpledialog.askinteger("Set k2", "Enter k2:", initialvalue=self.k2)

    def set_threshold(self):
        self.threshold = simpledialog.askfloat("Set Threshold", "Enter threshold:", initialvalue=self.threshold)

    def start_camera(self):
        if self.faiss_index is None:
            tk.Label(self.root, text="Load faces first!").pack()
            return
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                face_image = frame[y:y+h, x:x+w]
                face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                inputs = processor(images=face_rgb, return_tensors="pt")
                with torch.no_grad():
                    outputs = model.get_image_features(**inputs)
                face_embedding = outputs.cpu().numpy().flatten()
                face_embedding /= np.linalg.norm(face_embedding)
                majority_class, class_counts = classify_face(face_embedding, self.faiss_index, known_face_names, self.k1, self.k2, self.threshold)
                match_text = f"{majority_class} ({class_counts[majority_class]} votes)"
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, match_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def quit_app(self):
        self.root.destroy()

if __name__ == "__main__":
    app = FaceRecognitionApp()
