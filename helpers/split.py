import os
import random
import shutil

def split_subfolders(base_path, train_ratio=0.8):
    """
    Randomly splits images in each subfolder into train/val folders.
    
    Args:
        base_path (str): Path to the main directory (e.g., 'video_faces').
        train_ratio (float): Ratio of images to include in the train folder. Default is 0.8 (80%).
    """
    # Get list of subfolders in the base directory
    subfolders = [os.path.join(base_path, subfolder) for subfolder in os.listdir(base_path)
                  if os.path.isdir(os.path.join(base_path, subfolder))]

    for subfolder in subfolders:
        images = [img for img in os.listdir(subfolder) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)  # Shuffle the images randomly
        
        train_count = int(len(images) * train_ratio)
        
        # Create train/val subdirectories
        train_folder = os.path.join(subfolder, "train")
        val_folder = os.path.join(subfolder, "val")
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)

        # Move images to train and val folders
        for i, img in enumerate(images):
            src_path = os.path.join(subfolder, img)
            if i < train_count:
                dest_path = os.path.join(train_folder, img)
            else:
                dest_path = os.path.join(val_folder, img)
            
            shutil.move(src_path, dest_path)

        print(f"Subfolder '{subfolder}' split into:")
        print(f"  Train: {len(os.listdir(train_folder))} images")
        print(f"  Val: {len(os.listdir(val_folder))} images")

if __name__ == "__main__":
    base_path = "C:/Users/Korisnik/Desktop/WORKING_CV_ATTENDANCE/helpers/ponedijak_popodne"
    split_subfolders(base_path)
