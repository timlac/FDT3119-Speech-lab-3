import torch
import os

class Trainer:

    def __init__(self, model, optimizer, path=None):
        self.model = model
        self.optimizer = optimizer
        self.path = path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model.to(self.device)


    def train(self, train_loader, val_loader=None, epochs=100):
        train_losses = []
        val_losses = []

        for epoch in range(epochs):

            self.model.train()
            total_loss = 0

            for X_batch, y_batch in train_loader:

                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                outputs, loss = self.model(X_batch, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            if val_loader:
                val_loss = self.validate(val_loader)
                val_losses.append(val_loss)
                print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}")

        return train_losses, val_losses

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                _, loss = self.model(X_batch, y_batch)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def predict(self, test_loader):
        self.model.eval()
        y_pred_list = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                y_pred, _ = self.model(X_batch)
                y_pred_list.append(y_pred.cpu())

        y_pred_tensor = torch.cat(y_pred_list, dim=0)
        return y_pred_tensor.numpy()

    def save_model(self):
        torch.save(self.model.state_dict(), self.path)
        print(f"Model saved at {self.path}")

    def load_model(self):
        self.model.load_state_dict(torch.load(self.path))
        self.model.to(self.device)
        print("Model loaded successfully!")

