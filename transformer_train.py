import pickle

import torch

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

import numpy as np

from transformer_model import TrajectoryTransformer

# Load data

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

# Dataset class

class TrajectoryDataset(Dataset):

    def __init__(self, samples):

        self.samples = samples

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        past   = torch.FloatTensor(sample['past'])

        future = torch.FloatTensor(sample['future'])

        return past, future

# Create datasets and dataloaders

train_dataset = TrajectoryDataset(train_samples)

val_dataset   = TrajectoryDataset(val_samples)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)

# Model, loss, optimizer

model     = TrajectoryTransformer()

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

EPOCHS = 50

best_val_loss = float('inf')

# Training loop

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

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(model.state_dict(), 'data/transformer_model_best.pth')

        print(f"  → Best model saved! Val Loss: {val_loss:.4f}")

# Save final model

torch.save(model.state_dict(), 'data/transformer_model.pth')

print("\nModel saved to data/transformer_model.pth")