"""Концепция 47: Early stopping.

Останавливаем обучение, когда ошибка на валидации перестаёт падать.
Тоже форма регуляризации - не даём модели переучиться.
"""
import numpy as np

rng = np.random.default_rng(0)
N = 60
X = rng.normal(size=(N, 1))
y_true = X[:, 0] ** 3
y = y_true + rng.normal(scale=0.3, size=N)
split = 40
Xtr, ytr, Xva, yva = X[:split], y[:split], X[split:], y[split:]


def design(x, deg):
    return np.hstack([x ** k for k in range(deg + 1)])


best_val, best_deg, patience = 1e9, None, 3
bad = 0
for deg in range(1, 20):
    Atr, Ava = design(Xtr, deg), design(Xva, deg)
    w, *_ = np.linalg.lstsq(Atr, ytr, rcond=None)
    val_mse = ((Ava @ w - yva) ** 2).mean()
    print(f"deg={deg:>2}: val MSE={val_mse:.3f}")
    if val_mse < best_val - 1e-3:
        best_val, best_deg, bad = val_mse, deg, 0
    else:
        bad += 1
        if bad >= patience:
            print(f"--> early stop. Лучший degree={best_deg}")
            break
