import numpy as np
import torch

class MajorityClassClassifier:
    def __init__(self):
        self.majority_class = None

    def fit(self, y_train):
        # y_train: torch tensor or numpy array of labels
        y = y_train.numpy() if isinstance(y_train, torch.Tensor) else y_train
        values, counts = np.unique(y, return_counts=True)
        self.majority_class = values[np.argmax(counts)]

    def predict(self, X):
        # X is only used to determine length
        n = len(X)
        return np.full(n, self.majority_class)

    def score(self, X, y_true):
        y_pred = self.predict(X)
        return np.mean(y_pred == y_true.numpy())