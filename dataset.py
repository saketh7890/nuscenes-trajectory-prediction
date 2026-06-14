import torch
from torch.utils.data import Dataset

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