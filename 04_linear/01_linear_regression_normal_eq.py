"""Концепция 36: Линейная регрессия через нормальное уравнение.

w = (X^T X)^-1 X^T y. Закрытое решение без итераций.
"""
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(50, 1))
y = 3 * X[:, 0] + 1 + rng.normal(scale=0.1, size=50)

# Добавляем столбец единиц для свободного коэффициента (bias)
X_b = np.hstack([np.ones((len(X), 1)), X])
w = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
print(f"Найдены коэффициенты: bias={w[0]:.3f}, slope={w[1]:.3f}")
print("Истинные:           bias=1.000, slope=3.000")
