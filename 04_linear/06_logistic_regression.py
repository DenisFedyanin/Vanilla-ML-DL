"""Концепция 40: Логистическая регрессия (бинарная классификация).

p = sigma(X w + b). Учится максимизацией log-likelihood = минимизацией log-loss.
"""
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 2))
y = (X[:, 0] + X[:, 1] > 0).astype(int)  # настоящая граница: x0+x1=0
Xb = np.hstack([np.ones((200, 1)), X])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


w = np.zeros(3)
lr = 0.1
for _ in range(2000):
    p = sigmoid(Xb @ w)
    grad = Xb.T @ (p - y) / len(y)
    w -= lr * grad

acc = ((sigmoid(Xb @ w) > 0.5) == y).mean()
print(f"Веса: {w}")
print(f"Accuracy на трейне: {acc:.3f}")
