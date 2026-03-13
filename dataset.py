import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Lambda

from utility.utils import get_dataloader


class AddExtraDimension:
    def __call__(self, tensor):
        return tensor.unsqueeze(0)


class SubsetEEG:
    def __init__(self, channel_indices):
        self.channel_indices = channel_indices

    def __call__(self, eeg_tensor):
        # eeg_tensor shape: (channels, time) or (C, T)
        return eeg_tensor[:, self.channel_indices, :]


class MultiModalEEGDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        eeg_column,
        target_column,
        exclude_cols,
        eeg_transform=None,
    ):

        # Load data from CSV file
        self.data = pd.read_csv(annotations_file)

        self.eeg_column = eeg_column
        self.target_column = target_column
        self.eeg_transform = eeg_transform
        self.exclude_cols = exclude_cols

        # Identify all feature columns excluding EEG and target columns
        self.feature_columns = [
            col
            for col in self.data.columns
            if col not in [eeg_column, target_column] + exclude_cols
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # Load EEG data from the file path
        eeg_path = self.data.iloc[idx][self.eeg_column]
        eeg_data = np.load(eeg_path)

        # Apply optional EEG transformation, if provided
        if self.eeg_transform:
            eeg_data = self.eeg_transform(eeg_data)

        # Load all feature columns into a tensor
        features = [self.data.iloc[idx][col] for col in self.feature_columns]
        feature_tensor = torch.tensor(features, dtype=torch.float32)

        # Load target column into a tensor
        target = self.data.iloc[idx][self.target_column]
        target_tensor = torch.tensor(target, dtype=torch.float32)

        return (eeg_data, feature_tensor), target_tensor
