import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt

from models.load_data import MFCCDataModule
from models.trainer import Trainer
from models.model import DNN


with open('../stateList.pkl', 'rb') as file:
    stateList = pickle.load(file)

output_dim = len(stateList)

data = np.load("../data/preprocessed/lmfcc_dynamic_standard.npz", allow_pickle=True)

data_module = MFCCDataModule(data, output_dim, batch_size=512)
train_loader, val_loader, test_loader = data_module.get_dataloaders()

input_dim = data_module.input_dim

dnn_model = DNN(input_dim, output_dim)

optimizer = torch.optim.Adam(dnn_model.parameters(), lr=1e-3, weight_decay=1e-4)

trainer = Trainer(model=dnn_model, optimizer=optimizer)
train_losses, val_losses, train_accuracies, val_accuracies = trainer.train(train_loader, val_loader, epochs=10)

# Evaluate on test set
avg_loss, accuracy, y_true, y_pred = trainer.evaluate(test_loader)
print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")


# Plot Loss
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
if val_losses:
    plt.plot(val_losses, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
if val_accuracies:
    plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training & Validation Accuracy")
plt.legend()

plt.tight_layout()
plt.show()