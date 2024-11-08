import torch
from transformers import CLIPModel, CLIPProcessor
import os

    
def load_clip_model(weights_path, dataset_path):
    """
    Loads the CLIP model and processor, initializes the classifier,
    and loads weights if the dataset contains known classes.
    """
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Determine number of classes from the dataset
    class_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    num_classes = len(class_dirs)

    # Initialize the classifier based on the number of classes
    classifier = torch.nn.Linear(model.config.projection_dim, num_classes if num_classes > 0 else 1)
    
    # Load classifier weights only if they match the current classifier dimensions
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path)
        
        # Check if the saved weights match the classifier's shape
        if state_dict['weight'].shape == classifier.weight.shape and state_dict['bias'].shape == classifier.bias.shape:
            classifier.load_state_dict(state_dict)
            print("Loaded classifier weights successfully.")
        else:
            print("Warning: Saved weights do not match the current classifier dimensions. Initializing with new weights.")
    else:
        print("No weights found. Initializing classifier with random weights.")
    
    model.eval()
    classifier.eval()
    return model, processor, classifier


    # Load the pre-trained CLIP model and processor
    

if __name__ == "__main__":
    # Define the paths...ovako mi dela lokalno
    weights_path = "C:/Users/Korisnik/Desktop/WORKING_CV_ATTENDANCE/fine_tuned_classifier.pth"  
    dataset_path = "C:/Users/Korisnik/Desktop/WORKING_CV_ATTENDANCE/known_faces"  
    # Load the model and processor
    model, processor, classifier = load_clip_model(weights_path, dataset_path)

    print("Model, processor, and classifier loaded successfully.")
