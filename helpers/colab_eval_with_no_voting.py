# Majority voting not implemented. Mixes up a few classes

import os
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Set the dataset path from Google Drive
base_dir = "/content/drive/MyDrive/Prvih100"  # Replace <folder_name> with your dataset folder name

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("CLIP model loaded successfully!")

# Data placeholders
known_face_encodings = []
known_face_names = []

# Add known face
def add_known_face(image_path, name):
    print(f"Processing image for known face: {image_path}")
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    embedding = outputs.cpu().numpy().flatten()
    known_face_encodings.append(embedding / np.linalg.norm(embedding))
    known_face_names.append(name)
    print(f"Added known face: {name}")

# Load known faces from 'train' folders
def load_known_faces(base_dir=base_dir):
    print("Loading known faces...")
    for class_name in os.listdir(base_dir):
        train_dir = os.path.join(base_dir, class_name, "train")
        if os.path.isdir(train_dir):
            print(f"Processing class: {class_name}")
            for image_file in os.listdir(train_dir):
                image_path = os.path.join(train_dir, image_file)
                add_known_face(image_path, class_name)
    print("All known faces loaded successfully!")

# Build FAISS index
def build_index(encodings):
    print("Building FAISS index...")
    dimension = encodings[0].shape[0]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(np.array(encodings))
    print("FAISS index built successfully!")
    return faiss_index

# Classify with FAISS
def classify_face(face_embedding, faiss_index, threshold=0.6):
    D, I = faiss_index.search(face_embedding[np.newaxis, :], k=1)
    if D[0][0] <= threshold:
        return known_face_names[I[0][0]]
    return "Unknown"

# Evaluate on validation set
def evaluate(val_dir, faiss_index):
    print("Evaluating on validation set...")
    true_labels = []
    pred_labels = []
    for class_name in os.listdir(val_dir):
        val_class_dir = os.path.join(val_dir, class_name, "val")
        if os.path.isdir(val_class_dir):
            print(f"Evaluating class: {class_name}")
            for image_file in os.listdir(val_class_dir):
                image_path = os.path.join(val_class_dir, image_file)
                print(f"Evaluating image: {image_path}")
                image = Image.open(image_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model.get_image_features(**inputs)
                face_embedding = outputs.cpu().numpy().flatten()
                face_embedding /= np.linalg.norm(face_embedding)
                pred_label = classify_face(face_embedding, faiss_index)
                true_labels.append(class_name)
                pred_labels.append(pred_label)
    print("Validation completed!")
    return true_labels, pred_labels

# Metrics and Visualization
def calculate_metrics(true_labels, pred_labels):
    print("Calculating metrics...")
    print("Classification Report:")
    print(classification_report(true_labels, pred_labels))
    cm = confusion_matrix(true_labels, pred_labels, labels=list(set(true_labels)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=list(set(true_labels)), yticklabels=list(set(true_labels)))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
    print("Metrics calculated and visualized!")

# Main Workflow
if __name__ == "__main__":
    # Load known faces and build FAISS index
    load_known_faces()
    faiss_index = build_index(known_face_encodings)

    # Evaluate on validation set
    val_dir = os.path.join(base_dir)  # Ensure this points to the base directory
    true_labels, pred_labels = evaluate(val_dir, faiss_index)

    # Calculate metrics
    calculate_metrics(true_labels, pred_labels)
