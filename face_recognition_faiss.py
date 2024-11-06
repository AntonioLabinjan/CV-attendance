import numpy as np
import faiss
from PIL import Image
import torch
import clip  # OpenAI's CLIP model

# Step 1: Load the images and extract embeddings using CLIP
# Initialize the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)  # You can choose different models such as ViT-B/32, RN50

# Image paths to compare
image_paths = ["/content/WIN_20241106_16_51_20_Pro.jpg", "/content/WIN_20241106_16_51_12_Pro.jpg"]

# Step 2: Extract embeddings for each image using CLIP
embeddings = []
for image_path in image_paths:
    # Open image and preprocess it for CLIP
    image = Image.open(image_path)
    image_input = preprocess(image).unsqueeze(0).to(device)
    
    # Get the image embedding from CLIP
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        
    # Normalize the embedding (important for similarity comparison)
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    
    embeddings.append(image_features.cpu().numpy().flatten())  # Convert to numpy array and flatten

# Convert embeddings to numpy array for FAISS
embeddings = np.array(embeddings).astype('float32')

# Step 3: Create FAISS index to store image embeddings
d = embeddings.shape[1]  # Number of dimensions in the embedding vector (e.g., 512 for ViT-B/32)
index = faiss.IndexFlatL2(d)  # L2 distance (Euclidean distance) for similarity comparison
index.add(embeddings)  # Add embeddings to the index
print(f"Number of embeddings in the index: {index.ntotal}")

# Step 4: Perform similarity search - find the most similar faces to a query face
# Let's use the first image as the query
query_embedding = embeddings[0].reshape(1, -1)  # Reshape the query embedding to 2D

k = 2  # We want to see the top 3 nearest neighbors (excluding the query image itself)
D, I = index.search(query_embedding, k)

# Step 5: Output the results
print("\nSimilarity search - Nearest neighbors to the first image:")
for i in range(k):
    print(f"\nQuery Image: {image_paths[0]}")
    print(f"Nearest neighbor index (I): {I[0][i]}")  # Indices of the nearest neighbors
    print(f"Distance (D): {D[0][i]}")  # Distance to the nearest neighbors
    print(f"Neighbor Image: {image_paths[I[0][i]]}")  # Path to the nearest neighbor image

# 12 => 51 .... 0.3787812888622284 (različiti ljudi)
# 51 => 12 .... 0.3787812888622284 (različiti ljudi)

# 20 => 20 .... 0 (jer je ista slika)
# 20 => 12 .... 0.08069279789924622 (isti čovik, ali različita slika)



