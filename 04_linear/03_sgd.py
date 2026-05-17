"""Концепция 38: Стохастический градиентный спуск (SGD).

На каждом шаге берём ОДИН случайный пример и обновляем веса по нему.
Быстрее по итерации, но 'шумнее'.
"""
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 1))
y = 2 * X[:, 0] + 0.5 + rng.normal(scale=0.1, size=100)
X_b = np.hstack([np.ones((100, 1)), X])

w = np.zeros(2)
lr = 0.05
for epoch in range(20):
    for i in rng.permutation(len(X_b)):
        xi = X_b[i]; yi = y[i]
        grad = 2 * xi * (xi @ w - yi)
        w -= lr * grad

print(f"Найдено: bias={w[0]:.3f}, slope={w[1]:.3f}")
