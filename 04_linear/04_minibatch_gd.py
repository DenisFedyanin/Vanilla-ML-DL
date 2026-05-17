"""Концепция 39: Mini-batch GD.

Берём батч размера B (например, 32). Компромисс между полным GD и SGD.
"""
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(500, 1))
y = -1.0 * X[:, 0] + 4 + rng.normal(scale=0.1, size=500)
X_b = np.hstack([np.ones((500, 1)), X])

w = np.zeros(2)
lr = 0.1
batch = 32
for epoch in range(50):
    idx = rng.permutation(len(X_b))
    for start in range(0, len(idx), batch):
        b = idx[start:start + batch]
        grad = 2 / len(b) * X_b[b].T @ (X_b[b] @ w - y[b])
        w -= lr * grad

print(f"Найдено: bias={w[0]:.3f}, slope={w[1]:.3f}  (истинные 4 и -1)")
