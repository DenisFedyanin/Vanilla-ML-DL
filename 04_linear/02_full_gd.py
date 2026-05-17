"""Концепция 37: Полный (batch) градиентный спуск.

На каждом шаге считаем градиент по ВСЕМ данным сразу.
Шаг: w = w - lr * grad.
"""
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 1))
y = 2 * X[:, 0] + 0.5 + rng.normal(scale=0.1, size=100)
X_b = np.hstack([np.ones((100, 1)), X])

w = np.zeros(2)
lr = 0.1
for it in range(200):
    grad = 2 / len(X_b) * X_b.T @ (X_b @ w - y)
    w -= lr * grad

print(f"Найдено: bias={w[0]:.3f}, slope={w[1]:.3f}  (истинные 0.5 и 2.0)")
