from torch.utils.data import DataLoader

import pickle
import torch
import matplotlib.pyplot as plt

from baseline.majority_classifier import MajorityClassClassifier
from models.load_data import MFCCDataModule
from models.trainer import Trainer
from models.model import DNN


with open('../stateList.pkl', 'rb') as file:
    stateList = pickle.load(file)

output_dim = len(stateList)

data_module = MFCCDataModule("../data/preprocessed/mspec_standard.npz", output_dim, batch_size=512)

# Load your dataset
train_set, val_set, test_set = data_module.get_datasets()
y_train = torch.tensor(train_set[:][1])
y_val = torch.tensor(val_set[:][1])

# Fit and evaluate
baseline = MajorityClassClassifier()
baseline.fit(y_train)

val_acc = baseline.score(val_set, y_val)
print(f"Validation accuracy (majority class baseline): {val_acc:.4f}")