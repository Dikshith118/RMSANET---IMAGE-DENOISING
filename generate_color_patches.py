import os
import numpy as np
from PIL import Image
from torchvision import transforms

def generate_patches(img_dir, patch_size, num_patches_per_image, save_path):
    """
    Generates and saves RGB image patches from a directory of full images.

    Args:
        img_dir (str): Directory containing the original images.
        patch_size (int): The size of the patches to crop.
        num_patches_per_image (int): Number of patches to generate from each image.
        save_path (str): The path to save the .npy file.
    """
    # Check for subfolder 'original', else use img_dir directly
    original_img_dir = os.path.join(img_dir, 'original')
    if os.path.exists(original_img_dir):
        image_files = [os.path.join(original_img_dir, f) 
                       for f in os.listdir(original_img_dir) 
                       if f.endswith(('.png', '.jpg', '.bmp'))]
    else:
        image_files = [os.path.join(img_dir, f) 
                       for f in os.listdir(img_dir) 
                       if f.endswith(('.png', '.jpg', '.bmp'))]
    
    all_patches = []

    # Define a simple transform for cropping
    crop_transform = transforms.RandomCrop(patch_size)

    print(f"Generating {num_patches_per_image} patches from each of {len(image_files)} images...")

    for img_path in image_files:
        try:
            # Load image as RGB (color)
            full_image = Image.open(img_path).convert('RGB')

            for _ in range(num_patches_per_image):
                patch = crop_transform(full_image)
                # Convert to NumPy array for storage
                all_patches.append(np.array(patch))
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    # Stack all patches into a single NumPy array
    all_patches_np = np.stack(all_patches, axis=0)  # shape: [num_patches, H, W, 3]

    # Save the array to a .npy file
    np.save(save_path, all_patches_np)
    print(f"Successfully saved {all_patches_np.shape[0]} patches to {save_path}")

if __name__ == '__main__':
    # --- Configuration for Patch Generation ---
    PATCH_SIZE = 50
    NUM_PATCHES_PER_IMAGE = 500  # Number of patches per image
    TRAIN_DATA_DIR = os.path.join('data', 'CBSD432')
    SAVE_FOLDER_PATH = os.path.join('data', 'trained_patches')
    SAVE_FILE_PATH = os.path.join(SAVE_FOLDER_PATH, 'train_color_patches.npy')
    
    # Check and create the save folder
    if not os.path.exists(SAVE_FOLDER_PATH):
        os.makedirs(SAVE_FOLDER_PATH)
    
    if os.path.exists(TRAIN_DATA_DIR):
        generate_patches(TRAIN_DATA_DIR, PATCH_SIZE, NUM_PATCHES_PER_IMAGE, SAVE_FILE_PATH)
    else:
        print(f"Error: Training data directory not found at {TRAIN_DATA_DIR}")
