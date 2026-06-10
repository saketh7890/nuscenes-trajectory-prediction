import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import TrajectoryPredictor

# Load data and stats
with open('data/val_samples.pkl', 'rb') as f:
    val_samples = pickle.load(f)

with open('data/stats.pkl', 'rb') as f:
    stats = pickle.load(f)

mean = stats['mean']
std  = stats['std']

# Load model
model = TrajectoryPredictor()
model.load_state_dict(torch.load('data/model.pth'))
model.eval()

# Pick 6 random samples to visualize
import random
indices = random.sample(range(len(val_samples)), 6)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for plot_idx, sample_idx in enumerate(indices):
    sample = val_samples[sample_idx]

    past   = torch.FloatTensor(sample['past']).unsqueeze(0)
    future = torch.FloatTensor(sample['future']).unsqueeze(0)

    with torch.no_grad():
        prediction = model(past)

    # Denormalize to meters
    past_m   = sample['past']   * std + mean
    future_m = sample['future'] * std + mean
    pred_m   = prediction.squeeze().numpy() * std + mean

    ax = axes[plot_idx]

    # Plot past trajectory
    ax.plot(past_m[:, 0], past_m[:, 1],
            'b-o', linewidth=2, markersize=6, label='Past')

    # Plot actual future
    ax.plot(future_m[:, 0], future_m[:, 1],
            'g-o', linewidth=2, markersize=6, label='Actual Future')

    # Plot predicted future
    ax.plot(pred_m[:, 0], pred_m[:, 1],
            'r--o', linewidth=2, markersize=6, label='Predicted Future')

    # Mark start and end
    ax.plot(past_m[0, 0], past_m[0, 1], 'bs', markersize=10)
    ax.plot(future_m[-1, 0], future_m[-1, 1], 'g*', markersize=12)

    # ADE for this sample
    diff = pred_m - future_m
    ade  = np.sqrt((diff ** 2).sum(axis=-1)).mean()

    ax.set_title(f"{sample['category']}\nADE: {ade:.2f}m", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

plt.suptitle('nuScenes Trajectory Prediction — LSTM Model\nBlue=Past  Green=Actual  Red=Predicted',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved predictions.png")