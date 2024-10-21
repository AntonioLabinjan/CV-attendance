import torch
from transformers import CLIPModel, CLIPProcessor
import os

def load_clip_model(weights_path, dataset_path):
    """
    Load the CLIP model, processor, and fine-tuned classifier.

    Parameters:
    - weights_path (str): Path to the fine-tuned classifier weights (.pth file).
    - dataset_path (str): Path to the dataset to determine the number of classes.

    Returns:
    - model (CLIPModel): Pre-trained CLIP model.
    - processor (CLIPProcessor): Processor for pre-processing images and text.
    - classifier (torch.nn.Linear): Fine-tuned classifier.
    """
    
    # Load the pre-trained CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Determine the number of classes based on the dataset
    num_classes = len(os.listdir(dataset_path))  # Each subdirectory is treated as a class

    # Initialize the classifier
    classifier = torch.nn.Linear(model.config.projection_dim, num_classes)

    # Load the fine-tuned classifier weights
    classifier.load_state_dict(torch.load(weights_path))
    
    # Set the model and classifier to evaluation mode
    model.eval()
    classifier.eval()
    
    return model, processor, classifier

if __name__ == "__main__":
    # Define the paths...ovako mi dela lokalno
    weights_path = "C:/Users/Korisnik/Desktop/WORKING_CV_ATTENDANCE/fine_tuned_classifier.pth"  
    dataset_path = "C:/Users/Korisnik/Desktop/WORKING_CV_ATTENDANCE/known_faces"  

    # Load the model and processor
    model, processor, classifier = load_clip_model(weights_path, dataset_path)

    print("Model, processor, and classifier loaded successfully.")
