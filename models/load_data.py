import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MFCCDataset(Dataset):
    def __init__(self, X, y, num_classes=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MFCCDataModule:
    def __init__(self, data, num_classes, batch_size=256):
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']
        self.X_test = data['X_test']
        self.y_test = data['y_test']

        self.num_classes = num_classes
        self.batch_size = batch_size

    @property
    def input_dim(self):
        return self.X_train.shape[1]

    def get_datasets(self):
        """Returns raw torch Dataset objects without DataLoaders"""
        train_set = MFCCDataset(self.X_train, self.y_train, self.num_classes)
        val_set = MFCCDataset(self.X_val, self.y_val, self.num_classes)
        test_set = MFCCDataset(self.X_test, self.y_test, self.num_classes)
        return train_set, val_set, test_set

    def get_dataloaders(self):
        train_set = MFCCDataset(self.X_train, self.y_train, self.num_classes)
        val_set = MFCCDataset(self.X_val, self.y_val, self.num_classes)
        test_set = MFCCDataset(self.X_test, self.y_test, self.num_classes)

        return (
            DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True),
            DataLoader(val_set, batch_size=self.batch_size),
            DataLoader(test_set, batch_size=self.batch_size),
        )
