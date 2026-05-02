import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps'

# Data
DATA_DIR = 'data/'
RESIZE = (140, 140)
CROP_SIZE = (128, 128)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Augmentation
HFLIP_PROB = 0.5
SATURATION_FACTOR = 1.3
COLOR_JITTER = (0.15, 0.1, 0.1, 0.05)

# Training
BATCH_SIZE = 64
LEARNING_RATE = 5e-5
TOTAL_EPOCHS = 256
KL_BETA = 1.0

# Loss
VGG_LOSS_WEIGHT = 10000

# Model
Z_DIM = 256
DROPOUT = 0.3

# Checkpoints
SAVE_DIR = "checkpoints"
RESUME_EPOCH = 128  # epoch number used to build the resume filename
DISP_EPOCH = 256    # epoch number used in disp_outs to load weights

# Display
N_SAMPLES = 10

# Weights
LOAD_WEIGHTS = True
SAVE_WEIGHTS = True