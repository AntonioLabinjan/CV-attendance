# OVAJ DELA. NA 100 LJUDI DAJE 100% ACCURACY

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
base_dir = "/content/drive/MyDrive/video_faces"  # Replace <folder_name> with your dataset folder name
#base_dir = "/content/known_faces"
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
def classify_face(face_embedding, faiss_index, known_face_names, k1=3, k2=5, threshold=0.6):
    """
    Classifies a face embedding using majority voting logic.
    Args:
        face_embedding: The embedding of the face to classify.
        faiss_index: FAISS index for known faces.
        known_face_names: List of known face names corresponding to FAISS index.
        k1: Number of nearest neighbors for majority voting.
        k2: Number of fallback neighbors.
        threshold: Similarity threshold for classification.
    Returns:
        majority_class: Predicted class label.
        class_counts: Vote counts for each class.
    """
    # Search the FAISS index for k2 nearest neighbors
    D, I = faiss_index.search(face_embedding[np.newaxis, :], k2)
    votes = {}

    # Gather votes from k1 nearest neighbors
    for idx, dist in zip(I[0][:k1], D[0][:k1]):
        if idx != -1 and dist <= threshold:
            label = known_face_names[idx]
            votes[label] = votes.get(label, 0) + 1

    # Fallback: Gather votes from k2 - k1 additional neighbors
    if not votes:
        for idx, dist in zip(I[0][k1:], D[0][k1:]):
            if idx != -1 and dist <= threshold:
                label = known_face_names[idx]
                votes[label] = votes.get(label, 0) + 1

    # Determine the majority class
    if votes:
        majority_class = max(votes, key=votes.get)
    else:
        majority_class = "Unknown"

    return majority_class, votes

# Evaluate on validation set
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
                # Pass the known_face_names argument
                pred_label, _ = classify_face(face_embedding, faiss_index, known_face_names)  
                true_labels.append(class_name)
                pred_labels.append(pred_label)
    print("Validation completed!")
    return true_labels, pred_labels



'''
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
'''
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



METRIKE:
Calculating metrics...
Classification Report:
              precision    recall  f1-score   support

           1       1.00      1.00      1.00         9
          10       1.00      1.00      1.00        10
          11       1.00      1.00      1.00         7
          12       1.00      1.00      1.00         9
          13       1.00      1.00      1.00         6
          14       1.00      1.00      1.00         2
          15       1.00      1.00      1.00         6
          16       1.00      1.00      1.00         5
          17       1.00      1.00      1.00         9
          18       1.00      1.00      1.00         5
          19       1.00      1.00      1.00         4
           2       1.00      1.00      1.00         9
          20       1.00      1.00      1.00         9
          21       1.00      1.00      1.00         5
          22       1.00      1.00      1.00        12
          23       1.00      1.00      1.00         3
          24       1.00      1.00      1.00         6
          25       1.00      1.00      1.00         5
          26       1.00      1.00      1.00         7
          27       1.00      1.00      1.00         7
          28       1.00      1.00      1.00        10
          29       1.00      1.00      1.00        18
           3       1.00      1.00      1.00         8
          30       1.00      1.00      1.00         7
          31       1.00      1.00      1.00         5
          32       1.00      1.00      1.00         5
          33       1.00      1.00      1.00         5
          34       1.00      1.00      1.00         5
          35       1.00      1.00      1.00         5
          36       1.00      1.00      1.00         5
          37       1.00      1.00      1.00         5
          38       1.00      1.00      1.00         5
          39       1.00      1.00      1.00         6
           4       1.00      1.00      1.00        14
          40       1.00      1.00      1.00         5
          41       1.00      1.00      1.00         5
          42       1.00      1.00      1.00        13
          43       1.00      1.00      1.00         5
          44       1.00      1.00      1.00         7
          45       1.00      1.00      1.00         8
          46       1.00      1.00      1.00         9
          47       1.00      1.00      1.00         6
          48       1.00      1.00      1.00        11
          49       1.00      1.00      1.00         5
           5       1.00      1.00      1.00         6
          50       1.00      1.00      1.00         8
          51       1.00      1.00      1.00         5
          52       1.00      1.00      1.00         5
          53       1.00      1.00      1.00         5
          54       1.00      1.00      1.00         5
          55       1.00      1.00      1.00         8
          56       1.00      1.00      1.00         6
          57       1.00      1.00      1.00         6
          58       1.00      1.00      1.00         5
          59       1.00      1.00      1.00         5
           6       1.00      1.00      1.00         7
          60       1.00      1.00      1.00         5
          61       1.00      1.00      1.00         5
          62       1.00      1.00      1.00         7
          63       1.00      1.00      1.00         6
          64       1.00      1.00      1.00         6
          65       1.00      1.00      1.00         5
          66       1.00      1.00      1.00         6
          67       1.00      1.00      1.00         5
          68       1.00      1.00      1.00         5
          69       1.00      1.00      1.00         7
           7       1.00      1.00      1.00         6
          70       1.00      1.00      1.00         7
          71       1.00      1.00      1.00         7
          72       1.00      1.00      1.00         5
          73       1.00      1.00      1.00         7
          74       1.00      1.00      1.00         5
          75       1.00      1.00      1.00         5
          76       1.00      1.00      1.00         4
          77       1.00      1.00      1.00         7
          78       1.00      1.00      1.00         5
          79       1.00      1.00      1.00         6
           8       1.00      1.00      1.00         7
          80       1.00      1.00      1.00         6
          81       1.00      1.00      1.00         9
          82       1.00      1.00      1.00         7
          83       1.00      1.00      1.00         5
          84       1.00      1.00      1.00         7
          85       1.00      1.00      1.00         6
          86       1.00      1.00      1.00         5
          87       1.00      1.00      1.00         5
          88       1.00      1.00      1.00         4
          89       1.00      1.00      1.00         5
           9       1.00      1.00      1.00         6
          90       1.00      1.00      1.00         7
          91       1.00      1.00      1.00        13
          92       1.00      1.00      1.00         3
          93       1.00      1.00      1.00         5
          94       1.00      1.00      1.00         6
          95       1.00      1.00      1.00         4
          96       1.00      1.00      1.00         5
          97       1.00      1.00      1.00         7
          98       1.00      1.00      1.00         5
          99       1.00      1.00      1.00         6

    accuracy                           1.00       637
   macro avg       1.00      1.00      1.00       637
weighted avg       1.00      1.00      1.00       637
