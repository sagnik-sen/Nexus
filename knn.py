# knn.py
import numpy as np
import pickle
import os
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = int(k)
        self.X_train = None
        self.y_train = None
    
    def _check_array(self, X):
        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def fit(self, X, y):
        X = self._check_array(X)
        self.X_train = X
        self.y_train = np.array(y)
        return self

    def _euclidean_distances(self, x):
        # x shape (n_features,) ; self.X_train shape (n_samples, n_features)
        return np.linalg.norm(self.X_train - x, axis=1)

    def _predict_single(self, x):
        dists = self._euclidean_distances(x)
        k_idx = np.argsort(dists)[:self.k]
        k_labels = self.y_train[k_idx]
        most_common = Counter(k_labels).most_common(1)[0][0]
        return most_common

    def predict(self, X):
        X = self._check_array(X)
        preds = [self._predict_single(x) for x in X]
        return np.array(preds)
    
    def accuracy(self, X, y):
        y = np.array(y)
        preds = self.predict(X)
        return np.mean(preds == y)

def save_model(model, path="models/knn_model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
