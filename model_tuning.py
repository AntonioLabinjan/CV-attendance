import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import matplotlib.pyplot as plt

# Define your dataset class to handle the custom dataset
class KnownFacesDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_names = sorted(os.listdir(data_root))  # Sort for consistent label assignment
        
        for label_idx, label_name in enumerate(self.label_names):
            label_dir = os.path.join(data_root, label_name)
            if os.path.isdir(label_dir):
                for img_name in os.listdir(label_dir):
                    img_path = os.path.join(label_dir, img_name)
                    if os.path.isfile(img_path):
                        self.image_paths.append(img_path)
                        self.labels.append(label_idx)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Define transformations for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

# Load your dataset and create a DataLoader
data_root = '/content/known_faces'  # Path where your dataset is located
dataset = KnownFacesDataset(data_root, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Load the CLIP model and processor from Hugging Face
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

# Freeze the parameters of the CLIP model (optional, depending on your fine-tuning strategy)
for param in model.parameters():
    param.requires_grad = False

# Add a new classification head (Linear layer) for your custom classes
classifier = torch.nn.Linear(model.config.projection_dim, len(os.listdir(data_root))).to(device)
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

# Lists to store the values for plotting
train_loss = []
train_acc = []

# Training loop
for epoch in range(120):  # Number of epochs can be adjusted
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        # Extract image features using the CLIP model
        with torch.no_grad():  # Do not calculate gradients for the pre-trained CLIP model
            outputs = model.get_image_features(pixel_values=images)
        
        # Pass the extracted features through the new classifier
        outputs = classifier(outputs)
        loss = loss_fn(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        total_loss += loss.item()
    
    # Calculate and store loss and accuracy for each epoch
    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    train_loss.append(avg_loss)
    train_acc.append(accuracy)
    
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

# Plotting loss and accuracy
epochs = range(1, 121)
plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'r', label='Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, 'b', label='Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()

# Save the fine-tuned classifier
torch.save(classifier.state_dict(), "fine_tuned_classifier.pth")
print("Fine-tuning completed and model saved.")
