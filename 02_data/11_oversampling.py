"""Концепция 23: Дисбаланс классов и oversampling.

Если класса 1 в 10 раз меньше класса 0, модель может научиться
всегда предсказывать 0. Простое решение: дублировать редкий класс (oversampling).
"""
import numpy as np

rng = np.random.default_rng(0)
X = np.arange(100).reshape(-1, 1).astype(float)
y = np.array([0] * 90 + [1] * 10)

idx_pos = np.where(y == 1)[0]
idx_neg = np.where(y == 0)[0]
# дублируем положительные с возвратом, пока не уравняем
idx_pos_up = rng.choice(idx_pos, size=len(idx_neg), replace=True)

X_bal = np.vstack([X[idx_neg], X[idx_pos_up]])
y_bal = np.concatenate([y[idx_neg], y[idx_pos_up]])

print("До:", np.bincount(y))
print("После oversampling:", np.bincount(y_bal))
