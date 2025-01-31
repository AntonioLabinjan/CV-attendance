import torch
import numpy as np
from model_loader import load_clip_model

weights_path = "C:/Users/Korisnik/Desktop/WORKING_CV_ATTENDANCE/fine_tuned_classifier.pth"
dataset_path = "C:/Users/Korisnik/Desktop/WORKING_CV_ATTENDANCE/known_faces"
model, processor, classifier = load_clip_model(weights_path, dataset_path)

# Sample image processing to get an embedding (adjust based on your model setup)
def get_sample_embedding():
    # Assuming `processor` and `model` are already defined
    dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)  # Create a dummy RGB image
    inputs = processor(images=dummy_image, return_tensors="pt")
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    embedding = outputs.cpu().numpy().flatten()
    embedding /= np.linalg.norm(embedding)
    return embedding

# Run to get dimension of a sample embedding
sample_embedding = get_sample_embedding()
print("Embedding dimension:", sample_embedding.shape[0])
