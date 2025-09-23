import numpy as np
from math import log10
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(img1, img2):
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Args:
        img1 (np.ndarray): The first image array (denoised image), with pixel values in [0, 1].
        img2 (np.ndarray): The second image array (clean image), with pixel values in [0, 1].
        
    Returns:
        float: The PSNR score. A higher value indicates better quality.
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100  # If there's no error, PSNR is considered infinite
    max_pixel = 1.0  # Pixel values are normalized between 0 and 1
    psnr = 20 * log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(img1, img2):
    """
    Calculates the Structural Similarity Index (SSIM) between two images.
    
    Args:
        img1 (np.ndarray): The first image array, with pixel values in [0, 1].
        img2 (np.ndarray): The second image array, with pixel values in [0, 1].
        
    Returns:
        float: The SSIM score. A value closer to 1 indicates higher similarity.
    """
    # Use channel_axis=-1 for color images
    # Optionally set win_size=3 if images are small
    return ssim(img1, img2, data_range=1.0, channel_axis=-1, win_size=3)
