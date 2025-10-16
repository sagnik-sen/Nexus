# test_models.py
import numpy as np
import pickle
import os

from linear_regression import LinearRegression, save_model as save_linear
from logistic_regression import LogisticRegression, save_model as save_logistic
from knn import KNN, save_model as save_knn

# small helper split function
def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    split = int(X.shape[0] * (1 - test_size))
    train_idx = idx[:split]
    test_idx = idx[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def make_regression(n_samples=200, n_features=1, noise=5.0, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    X = np.random.rand(n_samples, n_features) * 10  # uniform 0..10
    true_w = np.arange(1, n_features + 1) * 2.0
    y = X.dot(true_w) + 5.0 + np.random.normal(0, noise, size=n_samples)
    return X, y

def make_classification_blobs(n_samples=200, centers=2, n_features=2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    samples_per_center = n_samples // centers
    X = []
    y = []
    for i in range(centers):
        center = np.random.uniform(-5, 5, size=n_features)
        pts = center + np.random.randn(samples_per_center, n_features)
        X.append(pts)
        y.append(np.full(samples_per_center, i))
    X = np.vstack(X)
    y = np.concatenate(y)
    # shuffle
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]

def test_linear():
    print("=== Testing Linear Regression ===")
    X, y = make_regression(n_samples=300, n_features=1, noise=8.0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression(learning_rate=0.001, n_iterations=3000)
    model.fit(X_train, y_train)
    r2 = model.score(X_test, y_test)
    print("R2 on test:", r2)
    save_linear(model, "models/linear_model.pkl")

def test_logistic():
    print("\n=== Testing Logistic Regression ===")
    X, y = make_classification_blobs(n_samples=300, centers=2, n_features=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X_train, y_train)
    acc = model.accuracy(X_test, y_test)
    print("Accuracy on test:", acc)
    save_logistic(model, "models/logistic_model.pkl")

def test_knn():
    print("\n=== Testing KNN ===")
    X, y = make_classification_blobs(n_samples=300, centers=2, n_features=2, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    knn = KNN(k=5)
    knn.fit(X_train, y_train)
    acc = knn.accuracy(X_test, y_test)
    print("KNN accuracy on test:", acc)
    save_knn(knn, "models/knn_model.pkl")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    test_linear()
    test_logistic()
    test_knn()
    print("\nAll models saved in ./models/")
