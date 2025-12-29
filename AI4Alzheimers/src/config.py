import torch

# ==========================================
# 1. GRANDMASTER CONFIGURATION
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 384  # High Resolution
BATCH_SIZE = 16 
EPOCHS = 20     # Enough for OneCycle to converge
MAX_LR = 0.0005 # Peak Learning Rate
