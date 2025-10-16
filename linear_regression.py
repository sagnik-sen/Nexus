# linear_regression.py
import numpy as np
import pickle
import os

class LinearRegression:
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
    
    def fit(self, X, y):
        X = self._check_array(X)
        y = np.array(y, dtype=float)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        for i in range(self.n_iterations):
            y_pred = X.dot(self.weights) + self.bias
            error = y_pred - y
            dw = (1.0 / n_samples) * X.T.dot(error)
            db = (1.0 / n_samples) * np.sum(error)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            loss = np.mean(error ** 2)  # MSE
            self.loss_history.append(loss)
            if i % 100 == 0:
                print(f"[Linear] Iter {i:4d}  Loss: {loss:.6f}")
        return self
    
    def predict(self, X):
        X = self._check_array(X)
        return X.dot(self.weights) + self.bias
    
    def score(self, X, y):
        # R^2
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

# Save helper
def save_model(model, path="models/linear_model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
