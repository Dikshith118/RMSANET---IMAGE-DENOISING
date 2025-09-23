import torch
import torchvision
import os

def save_image(tensor, filename, folder='results'):
    """
    Saves a PyTorch tensor as an image file.

    Args:
        tensor (torch.Tensor): The image tensor to save.
        filename (str): The name of the file to save.
        folder (str, optional): The directory to save the image in. Defaults to 'results'.
    """
    # Ensure the folder exists
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Save the image
    save_path = os.path.join(folder, filename)
    torchvision.utils.save_image(tensor, save_path)
    print(f"Image saved to {save_path}")
