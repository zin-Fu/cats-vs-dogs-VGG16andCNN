import torch

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
RANDOM_SEED = 1
LEARNING_RATE = 0.001
NUM_EPOCHS = 3
BATCH_SIZE = 128

# Architecture
NUM_CLASSES = 2
