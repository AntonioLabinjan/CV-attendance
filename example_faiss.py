# random num test


import numpy as np
import faiss  # make faiss available

# Step 1: Generate sample data
d = 64                           # dimension of the vectors
nb = 100000                      # database size
nq = 10000                       # number of queries

np.random.seed(1234)             # ensure reproducibility
xb = np.random.random((nb, d)).astype('float32')  # database vectors
xb[:, 0] += np.arange(nb) / 1000.  # modify the first dimension slightly to create variance
xq = np.random.random((nq, d)).astype('float32')  # query vectors
xq[:, 0] += np.arange(nq) / 1000.  # similarly modify the first dimension of the query vectors

# Step 2: Create FAISS index
index = faiss.IndexFlatL2(d)   # L2 distance (Euclidean distance)
print("Is the index trained? ", index.is_trained)  # Check if the index is trained (for certain index types)

# Step 3: Add vectors to the index
index.add(xb)                  # Adding the database vectors to the index
print(f"Number of vectors in the index: {index.ntotal}")  # Display how many vectors are in the index

# Step 4: Perform a sanity check: searching for nearest neighbors of the first 5 database vectors
k = 4  # we want to see the 4 nearest neighbors
D, I = index.search(xb[:5], k)  # Search for the 4 nearest neighbors of the first 5 database vectors

# Step 5: Output the results of the sanity check
print("\nSanity check - Nearest neighbors of the first 5 database vectors:")
for i in range(5):
    print(f"\nQuery Vector {i+1}:")
    print(f"Nearest neighbors' indices (I): {I[i]}")  # Indices of the nearest neighbors
    print(f"Distances (D): {D[i]}")  # Distances to the nearest neighbors

# Step 6: Perform the actual search: searching for nearest neighbors of the query vectors
D, I = index.search(xq, k)  # Searching for the nearest neighbors of the query vectors

# Step 7: Output the results for some of the query vectors
print("\nActual search - Nearest neighbors of the first and last 5 query vectors:")

# Display results for the first 5 queries
print("\nNeighbors of the first 5 query vectors:")
for i in range(5):
    print(f"\nQuery Vector {i+1}:")
    print(f"Nearest neighbors' indices (I): {I[i]}")  # Indices of the nearest neighbors
    print(f"Distances (D): {D[i]}")  # Distances to the nearest neighbors

# Display results for the last 5 queries
print("\nNeighbors of the last 5 query vectors:")
for i in range(-5, 0):
    print(f"\nQuery Vector {nq+i+1}:")
    print(f"Nearest neighbors' indices (I): {I[i]}")  # Indices of the nearest neighbors
    print(f"Distances (D): {D[i]}")  # Distances to the nearest neighbors
