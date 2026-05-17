"""Концепция 103: Underfitting vs Overfitting.

Слишком простая модель - underfit (плохо и на трейне и на тесте).
Слишком сложная - overfit (хорошо на трейне, плохо на тесте).
"""
import numpy as np

rng = np.random.default_rng(0)
N = 50
X = rng.uniform(-3, 3, size=N)
y = np.sin(X) + 0.1 * rng.normal(size=N)
Xtr, ytr = X[:35], y[:35]
Xte, yte = X[35:], y[35:]

for deg in [1, 3, 15]:
    A_tr = np.vander(Xtr, deg + 1)
    A_te = np.vander(Xte, deg + 1)
    w, *_ = np.linalg.lstsq(A_tr, ytr, rcond=None)
    mse_tr = ((A_tr @ w - ytr) ** 2).mean()
    mse_te = ((A_te @ w - yte) ** 2).mean()
    print(f"degree {deg:>2}: MSE train={mse_tr:.3f}  test={mse_te:.3f}")
print("degree=1 - underfit, degree=15 - overfit.")
