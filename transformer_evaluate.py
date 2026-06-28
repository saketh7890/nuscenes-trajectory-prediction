import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import TrajectoryDataset
from transformer_model import TrajectoryTransformer

with open('data/val_samples.pkl', 'rb') as f:
    val_samples = pickle.load(f)
with open('data/stats.pkl', 'rb') as f:
    stats = pickle.load(f)

mean = stats['mean']
std  = stats['std']

val_dataset = TrajectoryDataset(val_samples)
val_loader  = DataLoader(val_dataset, batch_size=32, shuffle=False)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

model = TrajectoryTransformer().to(device)
model.load_state_dict(torch.load('data/transformer_model_best.pth', map_location=device))
model.eval()

total_ade = 0
count = 0

with torch.no_grad():
    for past, future, map_features, neighbor_features in val_loader:
        past = past.to(device)
        future = future.to(device)
        map_features = map_features.to(device)
        neighbor_features = neighbor_features.to(device)

        prediction = model(past, map_features, neighbor_features)
        pred_m   = prediction.cpu().numpy() * std + mean
        future_m = future.cpu().numpy() * std + mean
        future_m = future_m[:, np.newaxis, :, :]

        diff = pred_m - future_m
        dist = np.sqrt((diff ** 2).sum(axis=-1)).mean(axis=-1)
        min_ade = dist.min(axis=-1)

        total_ade += min_ade.sum()
        count += len(past)

minADE = total_ade / count
print(f"Transformer minADE@5: {minADE:.4f} meters")
print(f"Evaluated on {count} samples")
