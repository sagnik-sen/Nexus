# AIML Basics — Nexus Club Assignment

## 📘 Problem Statement
Implement three basic machine learning algorithms **from scratch** using NumPy and Pandas:
1. Linear Regression
2. Logistic Regression
3. K-Nearest Neighbors (KNN)

### Linear Regression
- Uses gradient descent to minimize Mean Squared Error.
- Parameters updated using:
  - dw = (1/n) * Xᵀ(Xw + b − y)
  - db = (1/n) * Σ(Xw + b − y)

### Logistic Regression
- Uses sigmoid activation for binary classification.
- Loss: Binary Cross-Entropy.
- Parameters updated with gradient descent.

### K-Nearest Neighbors (KNN)
- Non-parametric lazy learner.
- Classifies based on majority vote of nearest `k` points (using Euclidean distance).
