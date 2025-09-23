import torch
from torch.utils.data import DataLoader
from data.dataset import DenoisingDataset
from models.rmsanet import RMSANet
from utils.metrics import calculate_psnr, calculate_ssim
from utils.utils import save_image
import os
import numpy as np
import argparse

# --- 1. IMPORT CONFIGURATION ---
import config as cfg

parser = argparse.ArgumentParser(description='Evaluate RMSANet on a test dataset.')
parser.add_argument('--dataset', type=str, default='BSD68', help='Name of the test dataset folder inside data/')
parser.add_argument('--noise_sigma', type=int, default=25, help='Noise level to add to the images')
args = parser.parse_args()

# Corrected path construction for the test images
# It now points to the original images for the selected dataset
DATA_DIR = os.path.join('data', args.dataset)

# CRITICAL: Path now points to the color model
MODEL_PATH = os.path.join(cfg.CHECKPOINT_DIR, 'model_gary_best.pth')
NOISE_SIGMA = args.noise_sigma
RESULTS_DIR = os.path.join(cfg.RESULTS_DIR, f'noise_{NOISE_SIGMA}', args.dataset, 'grayscale_model')

# --- 2. SETUP ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = RMSANet(in_channels=cfg.IN_CHANNELS, out_channels=cfg.OUT_CHANNELS).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Create the test dataset and dataloader
# CRITICAL: We now use the is_training=False flag and pass the data path
test_dataset = DenoisingDataset(data_path=DATA_DIR, noise_sigma_min=NOISE_SIGMA, noise_sigma_max=NOISE_SIGMA, is_training=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# --- 3. EVALUATION AND SAVING ---
total_psnr = 0.0
total_ssim = 0.0
image_count = 0

print(f"Starting evaluation on {args.dataset} with Noise Sigma: {NOISE_SIGMA}...")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

with torch.no_grad():
    for noisy_image, clean_image in test_loader:
        noisy_image, clean_image = noisy_image.to(device), clean_image.to(device)
        
        denoised_image, _ = model(noisy_image)

        filename_base = f"image_{image_count}"
        save_image(noisy_image.squeeze(), f"{filename_base}_noisy.png", folder=RESULTS_DIR)
        save_image(clean_image.squeeze(), f"{filename_base}_clean.png", folder=RESULTS_DIR)
        save_image(denoised_image.squeeze(), f"{filename_base}_denoised.png", folder=RESULTS_DIR)
        
        denoised_np = denoised_image.squeeze().cpu().numpy()
        clean_np = clean_image.squeeze().cpu().numpy()

        psnr = calculate_psnr(denoised_np, clean_np)
        ssim_val = calculate_ssim(denoised_np, clean_np)

        total_psnr += psnr
        total_ssim += ssim_val
        image_count += 1

# --- 4. PRINT RESULTS ---
avg_psnr = total_psnr / image_count
avg_ssim = total_ssim / image_count

print(f"\n--- Evaluation Results ({args.dataset} with Noise Sigma: {NOISE_SIGMA}) ---")
print(f"Average PSNR: {avg_psnr:.4f}")
print(f"Average SSIM: {avg_ssim:.4f}")
print("Evaluation finished.")
