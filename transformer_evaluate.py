import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import TrajectoryDataset
from transformer_model import TrajectoryTransformer

# Load data
with open('data/val_samples.pkl', 'rb') as f:
    val_samples = pickle.load(f)

with open('data/stats.pkl', 'rb') as f:
    stats = pickle.load(f)

mean = stats['mean']
std  = stats['std']

# Dataset and dataloader
val_dataset = TrajectoryDataset(val_samples)
val_loader  = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load model
model = TrajectoryTransformer()
model.load_state_dict(torch.load('data/transformer_model.pth'))
model.eval()

# Evaluate
total_ade = 0
count = 0

with torch.no_grad():
    for past, future in val_loader:
        prediction = model(past)

        pred_m   = prediction.numpy() * std + mean
        future_m = future.numpy()     * std + mean

        diff = pred_m - future_m
        dist = np.sqrt((diff ** 2).sum(axis=-1))
        ade  = dist.mean(axis=-1)

        total_ade += ade.sum()
        count += len(past)

minADE = total_ade / count
print(f"Transformer minADE: {minADE:.4f} meters")
print(f"Evaluated on {count} samples")