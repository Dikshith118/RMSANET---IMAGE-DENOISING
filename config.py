import os

# --- General Configuration ---
PROJECT_NAME = "RMSANet_Patch_Training_Color_Noise_Range"
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"

# --- Training Parameters ---
TRAIN_PATCH_PATH = os.path.join("data", "trained_patches", "train_color_patches.npy")
VAL_SPLIT = 0.1 # 10% of patches for validation
NOISE_SIGMA_MIN = 0
NOISE_SIGMA_MAX = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 40
SAVE_FREQ = 5 # Save a regular checkpoint every N epochs
PATCH_SIZE = 50 # Patches are now 50x50

# --- Model Parameters ---
IN_CHANNELS = 1
OUT_CHANNELS = 1
NUM_RDB_BLOCKS = 4
