import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from data.dataset import DenoisingDataset
from models.rmsanet import RMSANet
import os
import time
import numpy as np

# --- 1. IMPORT CONFIGURATION ---
import config as cfg

# --- 2. SETUP ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create dataset from the saved patches file
full_dataset = DenoisingDataset(
    npy_path=cfg.TRAIN_PATCH_PATH,
    noise_sigma_min=cfg.NOISE_SIGMA_MIN,
    noise_sigma_max=cfg.NOISE_SIGMA_MAX
)
train_size = int((1 - cfg.VAL_SPLIT) * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(dataset=val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)

model = RMSANet(in_channels=cfg.IN_CHANNELS, out_channels=cfg.OUT_CHANNELS).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

if not os.path.exists(cfg.CHECKPOINT_DIR):
    os.makedirs(cfg.CHECKPOINT_DIR)

best_val_loss = float('inf')

# --- 3. TRAINING LOOP ---
print("Starting training on image patches with a range of noise levels...")
for epoch in range(cfg.NUM_EPOCHS):
    epoch_start_time = time.time()
    model.train()
    running_loss = 0.0
    for i, (noisy_image, clean_image) in enumerate(train_loader):
        noisy_image, clean_image = noisy_image.to(device), clean_image.to(device)
        optimizer.zero_grad()
        denoised_image, predicted_noise = model(noisy_image)
        true_noise = noisy_image - clean_image
        loss = criterion(predicted_noise, true_noise)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for noisy_image, clean_image in val_loader:
            noisy_image, clean_image = noisy_image.to(device), clean_image.to(device)
            denoised_image, predicted_noise = model(noisy_image)
            true_noise = noisy_image - clean_image
            val_loss += criterion(predicted_noise, true_noise).item()
    
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    
    avg_train_loss = running_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    
    print(f"Epoch [{epoch+1}/{cfg.NUM_EPOCHS}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Time: {epoch_duration:.2f}s")
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model_path = os.path.join(cfg.CHECKPOINT_DIR, 'model_color_best.pth')
        torch.save(model.state_dict(), model_path)
        print("Validation loss improved. Best model saved to checkpoints\model_color_best.pth")

    if (epoch + 1) % cfg.SAVE_FREQ == 0:
        model_path = os.path.join(cfg.CHECKPOINT_DIR, f'model_color_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Regular checkpoint saved to {model_path}")

print("Training finished.")
