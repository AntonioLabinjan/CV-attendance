# UPDATE: NE GLEDA SE ACCURACY, NEGO LOSS, JER JE ACCURACY MISLEADING
# Ovo ni baš dobro
# Ono prije ni valjalo jer doslovno nisan validira model, nego san gleda samo običan accuracy
#Epoch 120, Loss: 0.0852, Accuracy: 82.86%, Val Loss: 0.1954, Val Accuracy: 77.78%
#https://colab.research.google.com/drive/1qRAu902U1vWxUgDKnpomD5diilk3OPSF?authuser=2#scrollTo=I_yIS_eA40QL
# Moran složit bolji dataset (više klasa i instanci)
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import matplotlib.pyplot as plt
import torch.nn as nn

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

# Load your dataset
data_root = '/content/known_faces'  # Path where your dataset is located
dataset = KnownFacesDataset(data_root, transform=transform)

# Split the dataset into training and testing (80% train, 20% test)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the CLIP model and processor from Hugging Face
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

# Freeze the parameters of the CLIP model (optional, depending on your fine-tuning strategy)
for param in model.parameters():
    param.requires_grad = False

# Add a new classification head (Linear layer) with Dropout for your custom classes
class ClassifierWithDropout(nn.Module):
    def __init__(self, input_dim, num_classes, dropout_rate=0.5):
        super(ClassifierWithDropout, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        return x

classifier = ClassifierWithDropout(model.config.projection_dim, len(os.listdir(data_root)), dropout_rate=0.5).to(device)

optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

# Lists to store the values for plotting
train_loss = []
train_acc = []
val_loss = []
val_acc = []

# Function to evaluate the model on the test set
def evaluate(model, classifier, dataloader):
    model.eval()
    classifier.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Extract image features using the CLIP model
            outputs = model.get_image_features(pixel_values=images)

            # Pass the extracted features through the classifier
            outputs = classifier(outputs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# Training loop
for epoch in range(120):  # Number of epochs can be adjusted
    total_loss = 0
    correct = 0
    total = 0
    model.train()
    classifier.train()

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Extract image features using the CLIP model
        with torch.no_grad():  # Do not calculate gradients for the pre-trained CLIP model
            outputs = model.get_image_features(pixel_values=images)

        # Pass the extracted features through the classifier
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

    # Calculate and store loss and accuracy for each epoch (train)
    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    train_loss.append(avg_loss)
    train_acc.append(accuracy)

    # Validation
    val_avg_loss, val_accuracy = evaluate(model, classifier, test_loader)
    val_loss.append(val_avg_loss)
    val_acc.append(val_accuracy)

    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, Val Loss: {val_avg_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

# Plotting loss and accuracy
epochs = range(1, 121)
plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'g', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'orange', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()

# Save the fine-tuned classifier
torch.save(classifier.state_dict(), "fine_tuned_classifier_with_dropout.pth")
print("Fine-tuning completed and model saved.")

