import os
import cv2
import numpy as np
from imgaug import augmenters as iaa

# Define augmentation pipeline
augment_pipeline = iaa.Sequential([
    iaa.Fliplr(0.5),  # Flip 50% of the images horizontally
    iaa.Flipud(0.2),  # Flip 20% of the images vertically
    iaa.Crop(percent=(0, 0.1)),  # Randomly crop up to 10% of the image
    iaa.Multiply((0.8, 1.2)),  # Random brightness adjustment between 80% and 120%
    iaa.Affine(
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},  # Small translations
        rotate=(-30, 30)  # Slight rotation within ±30 degrees
    )
])

def count_images_in_subfolders(root_folder):
    """Count the number of images in each subfolder."""
    image_counts = {}
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)
        if os.path.isdir(subfolder_path):
            image_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            image_counts[subfolder] = len(image_files)
    return image_counts

def augment_images(subfolder_path, target_count):
    """Augment images in a subfolder until the target count is reached."""
    image_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_paths = [os.path.join(subfolder_path, f) for f in image_files]
    current_count = len(image_files)
    
    print(f"Augmenting {subfolder_path}: Current count = {current_count}, Target count = {target_count}")

    while current_count < target_count:
        # Select a random image to augment
        img_path = np.random.choice(image_paths)
        image = cv2.imread(img_path)

        if image is None:
            print(f"Error reading image {img_path}, skipping...")
            continue

        # Apply augmentations
        augmented_image = augment_pipeline(image=image)

        # Save the augmented image with a unique name
        augmented_filename = f"aug_{current_count}.jpg"
        augmented_path = os.path.join(subfolder_path, augmented_filename)
        cv2.imwrite(augmented_path, augmented_image)

        current_count += 1

def balance_dataset(root_folder):
    """Balance the dataset by augmenting smaller subfolders to match the largest subfolder."""
    # Step 1: Count images in each subfolder
    image_counts = count_images_in_subfolders(root_folder)
    max_images = max(image_counts.values())
    print(f"Maximum number of images in a subfolder: {max_images}")

    # Step 2: Augment smaller subfolders
    for subfolder, count in image_counts.items():
        subfolder_path = os.path.join(root_folder, subfolder)
        if count < max_images:
            augment_images(subfolder_path, max_images)
        else:
            print(f"Subfolder {subfolder} already has {count} images. No augmentation needed.")

if __name__ == "__main__":
    root_folder = "C:/Users/Korisnik/Desktop/WORKING_CV_ATTENDANCE/helpers/video_faces"  # Replace with the path to your dataset folder
    balance_dataset(root_folder)
