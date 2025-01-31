import os
import shutil

# Paths to the dataset folders
train_dir = "C:/Users/Korisnik/Downloads/archive (4)/celeba/custom_celeba_500/train"
test_dir = "C:/Users/Korisnik/Downloads/archive (4)/celeba/custom_celeba_500/test"
val_dir = "C:/Users/Korisnik/Downloads/archive (4)/celeba/custom_celeba_500/val"

# Output directory where all images will be merged
output_dir = "C:/Users/Korisnik/Downloads/archive (4)/celeba/custom_celeba_500/merged"
os.makedirs(output_dir, exist_ok=True)

# Function to merge images
def merge_images(source_dirs, target_dir):
    for source_dir in source_dirs:
        # Iterate over all subfolders in the source directory
        for person_folder in os.listdir(source_dir):
            person_path = os.path.join(source_dir, person_folder)
            if os.path.isdir(person_path):
                # Create the target folder for the person if it doesn't exist
                target_person_folder = os.path.join(target_dir, person_folder)
                os.makedirs(target_person_folder, exist_ok=True)
                
                # Move all images to the target folder
                for image_file in os.listdir(person_path):
                    source_image_path = os.path.join(person_path, image_file)
                    target_image_path = os.path.join(target_person_folder, image_file)
                    
                    # Avoid overwriting if the same file exists
                    if not os.path.exists(target_image_path):
                        shutil.copy(source_image_path, target_image_path)

# Merge images from train, test, and val folders
merge_images([train_dir, test_dir, val_dir], output_dir)

print(f"Images merged into {output_dir}")
