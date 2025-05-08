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
        train_accuracies = []
        val_accuracies = []

        for epoch in range(epochs):

            self.model.train()
            total_loss = 0
            correct = 0
            total = 0

            for X_batch, y_batch in train_loader:

                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                outputs, loss = self.model(X_batch, y_batch)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                preds = outputs.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)


            avg_train_loss = total_loss / len(train_loader)
            train_accuracy = correct / total

            train_losses.append(avg_train_loss)
            train_accuracies.append(train_accuracy)

            if val_loader:
                val_loss, val_accuracy = self.validate(val_loader)
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
                print(f"Epoch [{epoch + 1}/{epochs}] - "
                      f"Train Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.4f} - "
                      f"Val Loss: {val_loss:.4f}, Acc: {val_accuracy:.4f}")
            else:
                print(f"Epoch [{epoch + 1}/{epochs}] - "
                      f"Train Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.4f}")

        return train_losses, val_losses, train_accuracies, val_accuracies

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs, loss = self.model(X_batch, y_batch)
                total_loss += loss.item()

                preds = outputs.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        return avg_loss, accuracy

    def evaluate(self, data_loader):
        """
        Evaluates the model on a dataset.

        Returns:
            loss (float): average loss
            accuracy (float): overall accuracy
            y_true (np.ndarray): ground-truth labels
            y_pred (np.ndarray): predicted labels
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        y_true = []
        y_pred = []

        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs, loss = self.model(X_batch, y_batch)
                preds = outputs.argmax(dim=1)

                total_loss += loss.item()
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

                y_true.append(y_batch.cpu())
                y_pred.append(preds.cpu())

        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total
        y_true = torch.cat(y_true).numpy()
        y_pred = torch.cat(y_pred).numpy()

        return avg_loss, accuracy, y_true, y_pred

    def predict(self, data_loader):
        """
        Predicts the labels for a dataset.

        Returns:
            y_pred (np.ndarray): predicted labels
        """
        self.model.eval()
        y_pred_list = []

        with torch.no_grad():
            for X_batch, _ in data_loader:
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

