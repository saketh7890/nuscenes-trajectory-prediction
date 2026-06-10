import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from model import TrajectoryPredictor

# Load pre-split samples
with open('data/train_samples.pkl', 'rb') as f:
    train_samples = pickle.load(f)

with open('data/val_samples.pkl', 'rb') as f:
    val_samples = pickle.load(f)

print(f"Train samples: {len(train_samples)}")
print(f"Val   samples: {len(val_samples)}")

class TrajectoryDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        past = torch.FloatTensor(sample['past'])
        future = torch.FloatTensor(sample['future'])
        return past, future

train_dataset = TrajectoryDataset(train_samples)
val_dataset   = TrajectoryDataset(val_samples)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)

model     = TrajectoryPredictor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 50

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for past, future in train_loader:
        optimizer.zero_grad()
        prediction = model(past)
        loss = criterion(prediction, future)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for past, future in val_loader:
            prediction = model(past)
            loss = criterion(prediction, future)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1:2d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

torch.save(model.state_dict(), 'data/model.pth')
print("\nModel saved to data/model.pth")