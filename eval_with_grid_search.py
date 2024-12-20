# ROC, AUC (ča je to?)
# GRID SEARCH
# BAYESIAN HYPEROPT

!pip install transformers 
!pip install faiss-cpu  # Use faiss-gpu if you have a GPU in your environment
!pip install faiss-gpu
!pip install scikit-learn
!pip install matplotlib
!pip install seaborn
!pip install pillow

# ako mi nestanu oni fajlovi, iman ih u downloadsima
import os
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pickle

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Set the dataset path from Google Drive
base_dir = "/content/drive/MyDrive/oldest_faces"  # Replace with your dataset folder name

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("CLIP model loaded successfully!")

# Data placeholders
known_face_encodings = []
known_face_names = []

# Save the known face encodings and FAISS index
def save_known_faces_and_index():
    with open('known_face_encodings.pkl', 'wb') as f:
        pickle.dump(known_face_encodings, f)
    with open('known_face_names.pkl', 'wb') as f:
        pickle.dump(known_face_names, f)
    faiss.write_index(faiss_index, 'faiss_index.index')
    print("Known faces and FAISS index saved!")

# Load the known face encodings and FAISS index
def load_known_faces_and_index():
    global known_face_encodings, known_face_names, faiss_index
    if os.path.exists('known_face_encodings.pkl') and os.path.exists('faiss_index.index'):
        with open('known_face_encodings.pkl', 'rb') as f:
            known_face_encodings = pickle.load(f)
        with open('known_face_names.pkl', 'rb') as f:
            known_face_names = pickle.load(f)
        faiss_index = faiss.read_index('faiss_index.index')
        print("Loaded known faces and FAISS index from disk!")
    else:
        print("No saved data found. Proceeding with processing faces...")
        load_known_faces() # This line is added to populate the lists if no saved data is found
        faiss_index = build_index(known_face_encodings)
        save_known_faces_and_index() 

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

from sklearn.metrics import accuracy_score
from itertools import product

def grid_search(val_dir, faiss_index, known_face_names, k1_values, k2_values, threshold_values):
    best_params = {}
    best_accuracy = 0
    results = []

    print("Starting grid search...")
    for k1, k2, threshold in product(k1_values, k2_values, threshold_values):
        print(f"\nTesting parameters: k1={k1}, k2={k2}, threshold={threshold}")
        
        # Evaluate with current parameters
        def classify_with_params(face_embedding):
            return classify_face(face_embedding, faiss_index, known_face_names, k1=k1, k2=k2, threshold=threshold)

        true_labels, pred_labels = evaluate(val_dir, faiss_index, classify_with_params)
        
        # Compute accuracy
        accuracy = accuracy_score(true_labels, pred_labels)
        print(f"Accuracy for k1={k1}, k2={k2}, threshold={threshold}: {accuracy:.4f}")
        results.append((k1, k2, threshold, accuracy))

        # Update best parameters
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {"k1": k1, "k2": k2, "threshold": threshold}

    print("\nGrid search completed.")
    print(f"Best parameters: {best_params}")
    print(f"Best accuracy: {best_accuracy:.4f}")

    return best_params, best_accuracy, results


# Classify with FAISS
'''
IGNORE OVO DOLE
#def classify_face(face_embedding, faiss_index, known_face_names, k1=10, k2=18, threshold=0.5): # ovo daje 69%
#def classify_face(face_embedding, faiss_index, known_face_names, k1=6, k2=8, threshold=0.75): # ovo daje 71%
IGNORE OVO GORE

def classify_face(face_embedding, faiss_index, known_face_names, k1=3, k2=5, threshold=4.50): # 72%
def classify_face(face_embedding, faiss_index, known_face_names, k1=3, k2=5, threshold=1.2): # 72%
#def classify_face(face_embedding, faiss_index, known_face_names, k1=3, k2=5, threshold=0.99): # 72%
#def classify_face(face_embedding, faiss_index, known_face_names, k1=3, k2=5, threshold=0.85): # 72%
#def classify_face(face_embedding, faiss_index, known_face_names, k1=3, k2=5, threshold=0.85): # 72%
def classify_face(face_embedding, faiss_index, known_face_names, k1=3, k2=5, threshold=0.70): # 72%
def classify_face(face_embedding, faiss_index, known_face_names, k1=3, k2=5, threshold=0.67): # 72%
def classify_face(face_embedding, faiss_index, known_face_names, k1=3, k2=5, threshold=0.65): # 72%
def classify_face(face_embedding, faiss_index, known_face_names, k1=3, k2=5, threshold=0.635): # 72%
def classify_face(face_embedding, faiss_index, known_face_names, k1=3, k2=5, threshold=0.62): # 72%
def classify_face(face_embedding, faiss_index, known_face_names, k1=3, k2=5, threshold=0.61): # 72%
#def classify_face(face_embedding, faiss_index, known_face_names, k1=3, k2=5, threshold=0.60): # 72% # OVO JE BILO NAJBOLJE NA NORMALNEMU DATASETU
#def classify_face(face_embedding, faiss_index, known_face_names, k1=3, k2=5, threshold=0.55): # 71%
#def classify_face(face_embedding, faiss_index, known_face_names, k1=3, k2=5, threshold=0.50): # 0.69
#def classify_face(face_embedding, faiss_index, known_face_names, k1=3, k2=5, threshold=0.45): # 0.65
#def classify_face(face_embedding, faiss_index, known_face_names, k1=3, k2=5, threshold=0.40): # 0.60
#def classify_face(face_embedding, faiss_index, known_face_names, k1=3, k2=5, threshold=0.20): # 0.13
#def classify_face(face_embedding, faiss_index, known_face_names, k1=3, k2=5, threshold=0.10): # 0.13

'''
# MISLIN DA ĆE NAS OGROMNI K-OVI SPASIT :); provat 20, 30 iz zajebancije i onda stvarno poć na grid
# provat 3,5,0.99 :)
#def classify_face(face_embedding, faiss_index, known_face_names, k1=5, k2=7, threshold=0.10): # 0.13
#def classify_face(face_embedding, faiss_index, known_face_names, k1=5, k2=7, threshold=0.20):# 0.24
#def classify_face(face_embedding, faiss_index, known_face_names, k1=5, k2=7, threshold=0.30): # 0.39
#def classify_face(face_embedding, faiss_index, known_face_names, k1=5, k2=7, threshold=0.40): # 0.60
#def classify_face(face_embedding, faiss_index, known_face_names, k1=5, k2=7, threshold=0.50): # 0.69
#def classify_face(face_embedding, faiss_index, known_face_names, k1=5, k2=7, threshold=0.60): # 0.71
#def classify_face(face_embedding, faiss_index, known_face_names, k1=5, k2=7, threshold=0.70): # 0.71
#def classify_face(face_embedding, faiss_index, known_face_names, k1=5, k2=7, threshold=0.80): # 0.71
#def classify_face(face_embedding, faiss_index, known_face_names, k1=5, k2=7, threshold=0.90): # 0.71
#def classify_face(face_embedding, faiss_index, known_face_names, k1=20, k2=30, threshold=0.99): # 0.70

def classify_face(face_embedding, faiss_index, known_face_names, k1=3, k2=5, threshold=0.60):
    D, I = faiss_index.search(face_embedding[np.newaxis, :], k2)
    votes = {}
    for idx, dist in zip(I[0][:k1], D[0][:k1]):
        if idx != -1 and dist <= threshold:
            label = known_face_names[idx]
            votes[label] = votes.get(label, 0) + 1
    if not votes:
        for idx, dist in zip(I[0][k1:], D[0][k1:]):
            if idx != -1 and dist <= threshold:
                label = known_face_names[idx]
                votes[label] = votes.get(label, 0) + 1
    majority_class = max(votes, key=votes.get) if votes else "Unknown"
    return majority_class, votes

# Evaluate on validation set
'''
def evaluate(val_dir, faiss_index):
    print("Evaluating on validation set...")
    true_labels = []
    pred_labels = []
    
    # Track total and processed images
    total_images = 0
    processed_images = 0
    
    for class_name in os.listdir(val_dir):
        val_class_dir = os.path.join(val_dir, class_name, "val")
        if os.path.isdir(val_class_dir):
            print(f"Processing class: {class_name}")
            for image_file in os.listdir(val_class_dir):
                total_images += 1
                image_path = os.path.join(val_class_dir, image_file)
                print(f"  - Loading image: {image_file} from class {class_name}")
                
                # Open and process the image
                try:
                    image = Image.open(image_path).convert("RGB")
                except Exception as e:
                    print(f"    [ERROR] Failed to load image {image_file}: {e}")
                    continue

                # Generate image embeddings
                inputs = processor(images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model.get_image_features(**inputs)
                face_embedding = outputs.cpu().numpy().flatten()
                
                # Normalize the embedding
                face_embedding /= np.linalg.norm(face_embedding)
                print(f"    - Generated face embedding for {image_file}")
                
                # Classify the face
                pred_label, _ = classify_face(face_embedding, faiss_index, known_face_names)
                print(f"    - Predicted label: {pred_label}")
                
                # Append labels
                true_labels.append(class_name)
                pred_labels.append(pred_label)
                
                processed_images += 1
    
    print(f"\nEvaluation completed: {processed_images}/{total_images} images processed.")
    return true_labels, pred_labels
'''
def evaluate(val_dir, faiss_index, classify_function):
    print("Evaluating on validation set...")
    true_labels = []
    pred_labels = []
    
    total_images = 0
    processed_images = 0
    
    for class_name in os.listdir(val_dir):
        val_class_dir = os.path.join(val_dir, class_name, "val")
        if os.path.isdir(val_class_dir):
            print(f"Processing class: {class_name}")
            for image_file in os.listdir(val_class_dir):
                total_images += 1
                image_path = os.path.join(val_class_dir, image_file)
                print(f"  - Loading image: {image_file} from class {class_name}")
                
                try:
                    image = Image.open(image_path).convert("RGB")
                except Exception as e:
                    print(f"    [ERROR] Failed to load image {image_file}: {e}")
                    continue

                inputs = processor(images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model.get_image_features(**inputs)
                face_embedding = outputs.cpu().numpy().flatten()
                
                face_embedding /= np.linalg.norm(face_embedding)
                print(f"    - Generated face embedding for {image_file}")
                
                pred_label, _ = classify_function(face_embedding)
                print(f"    - Predicted label: {pred_label}")
                
                true_labels.append(class_name)
                pred_labels.append(pred_label)
                
                processed_images += 1
    
    print(f"\nEvaluation completed: {processed_images}/{total_images} images processed.")
    return true_labels, pred_labels

# Metrics and Visualization
from sklearn.metrics import recall_score, f1_score

def calculate_metrics(true_labels, pred_labels):
    print("Calculating metrics...")
    print(classification_report(true_labels, pred_labels))

    
    f1 = f1_score(true_labels, pred_labels, average="weighted")  # Use "weighted" to handle class imbalance
    recall = recall_score(true_labels, pred_labels, average="weighted")
    
    print(f"\nF1-Score (Weighted): {f1:.4f}")
    print(f"Recall (Weighted): {recall:.4f}")

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
    # Load known faces and build FAISS index if not saved
    load_known_faces_and_index()
    if not known_face_encodings:  # If there are no known faces loaded from disk
        load_known_faces()
        faiss_index = build_index(known_face_encodings)
        save_known_faces_and_index()  # Save for future use

    # Default parameters for initial evaluation
    default_k1 = 3
    default_k2 = 5
    default_threshold = 0.60

    # Evaluate on validation set with default parameters
    val_dir = os.path.join(base_dir)  # Ensure this points to the base directory
    true_labels, pred_labels = evaluate(
        val_dir, 
        faiss_index, 
        lambda face_embedding: classify_face(
            face_embedding, faiss_index, known_face_names, default_k1, default_k2, default_threshold
        )
    )

    # Define parameter ranges
    k1_values = [3, 5]
    k2_values = [5, 7]
    threshold_values = [0.60, 0.70]

    # Run grid search
    best_params, best_accuracy, results = grid_search(
        val_dir=val_dir, 
        faiss_index=faiss_index, 
        known_face_names=known_face_names, 
        k1_values=k1_values, 
        k2_values=k2_values, 
        threshold_values=threshold_values
    )

    # Output results
    print(f"Best parameters found: {best_params}")
    print(f"Best accuracy achieved: {best_accuracy}")

    # Calculate metrics
    calculate_metrics(true_labels, pred_labels)

