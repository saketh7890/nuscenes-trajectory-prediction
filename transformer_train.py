import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformer_model import TrajectoryTransformer

with open('data/train_samples.pkl', 'rb') as f:
    train_samples = pickle.load(f)
with open('data/val_samples.pkl', 'rb') as f:
    val_samples = pickle.load(f)
with open('data/stats.pkl', 'rb') as f:
    stats = pickle.load(f)

mean = stats['mean']
std  = stats['std']

print(f"Train samples: {len(train_samples)}")
print(f"Val   samples: {len(val_samples)}")

# Device setup — use Mac GPU (MPS) if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class TrajectoryDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        return (torch.FloatTensor(s['past']),
                torch.FloatTensor(s['future']),
                torch.FloatTensor(s['map_features']))

def wta_loss(predictions, targets):
    targets_expanded = targets.unsqueeze(1).expand_as(predictions)
    losses = ((predictions - targets_expanded) ** 2).mean(dim=[2, 3])
    best_idx = losses.argmin(dim=1)
    batch_size = predictions.shape[0]
    best_losses = losses[torch.arange(batch_size), best_idx]
    return best_losses.mean()

train_dataset = TrajectoryDataset(train_samples)
val_dataset   = TrajectoryDataset(val_samples)
train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader    = DataLoader(val_dataset,   batch_size=32, shuffle=False)

model     = TrajectoryTransformer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100, eta_min=1e-6
)
EPOCHS = 100
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for past, future, map_features in train_loader:
        past = past.to(device)
        future = future.to(device)
        map_features = map_features.to(device)

        optimizer.zero_grad()
        prediction = model(past, map_features)
        loss = wta_loss(prediction, future)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for past, future, map_features in val_loader:
            past = past.to(device)
            future = future.to(device)
            map_features = map_features.to(device)

            prediction = model(past, map_features)
            loss = wta_loss(prediction, future)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    scheduler.step()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'data/transformer_model_best.pth')
        print(f"  → Best model saved! Val Loss: {val_loss:.4f}")

torch.save(model.state_dict(), 'data/transformer_model.pth')
print("\nModel saved to data/transformer_model.pth")