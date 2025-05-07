
import pickle
import torch
import matplotlib.pyplot as plt

from models.load_data import MFCCDataModule
from models.trainer import Trainer
from models.model import DNN


with open('../stateList.pkl', 'rb') as file:
    stateList = pickle.load(file)

output_dim = len(stateList)

data_module = MFCCDataModule("../data/preprocessed/lmfcc_standard.npz", output_dim, batch_size=512)
train_loader, val_loader, test_loader = data_module.get_dataloaders()

input_dim = data_module.input_dim

dnn_model = DNN(input_dim, output_dim)
optimizer = torch.optim.Adam(dnn_model.parameters(), lr=0.001, weight_decay=1e-4)

trainer = Trainer(model=dnn_model, optimizer=optimizer)
train_losses, val_losses = trainer.train(train_loader, val_loader, epochs=100)

plt.plot(train_losses, label='Train Loss')
if val_losses:
    plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.show()