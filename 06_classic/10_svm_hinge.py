"""Концепция 58: Линейный SVM через hinge loss.

L = max(0, 1 - y * (w*x + b)) + lambda * ||w||^2.
"""
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 2))
y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)

w = np.zeros(2); b = 0.0
lr = 0.01; lam = 0.01
for epoch in range(200):
    for i in rng.permutation(len(X)):
        margin = y[i] * (X[i] @ w + b)
        if margin >= 1:
            w -= lr * (2 * lam * w)
        else:
            w -= lr * (2 * lam * w - y[i] * X[i])
            b -= lr * (-y[i])

acc = (np.sign(X @ w + b) == y).mean()
print(f"SVM train acc: {acc:.3f},  w={w}, b={b:.3f}")
