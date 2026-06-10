import torch
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model import TrajectoryPredictor

# Load val samples — properly split by trajectory
with open('data/val_samples.pkl', 'rb') as f:
    val_samples = pickle.load(f)

with open('data/stats.pkl', 'rb') as f:
    stats = pickle.load(f)

mean = stats['mean']
std = stats['std']

print(f"Val samples: {len(val_samples)}")

class TrajectoryDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        return torch.FloatTensor(s['past']), torch.FloatTensor(s['future'])

val_dataset = TrajectoryDataset(val_samples)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load model
model = TrajectoryPredictor()
model.load_state_dict(torch.load('data/model.pth'))
model.eval()

total_ade = 0
count = 0

with torch.no_grad():
    for past, future in val_loader:
        prediction = model(past)

        # Denormalize to meters
        pred_m   = prediction.numpy() * std + mean
        future_m = future.numpy()     * std + mean

        # ADE in real meters
        diff = pred_m - future_m
        dist = np.sqrt((diff ** 2).sum(axis=-1))
        ade  = dist.mean(axis=-1)

        total_ade += ade.sum()
        count += len(past)

minADE = total_ade / count
print(f"minADE: {minADE:.4f} meters")
print(f"Evaluated on {count} samples")