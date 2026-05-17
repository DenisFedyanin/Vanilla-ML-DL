"""Концепция 45: L2 регуляризация (Ridge).

К потере добавляем lambda * ||w||^2. Веса 'сжимаются' к нулю.
Закрытое решение: w = (X^T X + lambda * I)^-1 X^T y.
"""
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(50, 3))
y = X[:, 0] * 2 + rng.normal(scale=0.5, size=50)

for lam in [0.0, 1.0, 100.0]:
    A = X.T @ X + lam * np.eye(X.shape[1])
    w = np.linalg.solve(A, X.T @ y)
    print(f"lambda={lam:>6}: weights={w}")
print("Большая lambda - меньше веса (модель проще).")
