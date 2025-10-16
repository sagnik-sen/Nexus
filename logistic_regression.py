# logistic_regression.py
import numpy as np
import pickle
import os

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def _check_array(self, X):
        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def _sigmoid(self, z):
        # stable sigmoid
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    
    def fit(self, X, y):
        X = self._check_array(X)
        y = np.array(y, dtype=float)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        eps = 1e-12
        for i in range(self.n_iterations):
            linear_model = X.dot(self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)

            error = y_pred - y
            dw = (1.0 / n_samples) * X.T.dot(error)
            db = (1.0 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Binary cross-entropy (stable)
            y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
            loss = -np.mean(y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped))
            self.loss_history.append(loss)

            if i % 100 == 0:
                print(f"[Logistic] Iter {i:4d}  Loss: {loss:.6f}")
        return self

    def predict_proba(self, X):
        X = self._check_array(X)
        linear_model = X.dot(self.weights) + self.bias
        return self._sigmoid(linear_model)
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def accuracy(self, X, y):
        y = np.array(y)
        preds = self.predict(X)
        return np.mean(preds == y)

def save_model(model, path="models/logistic_model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
