"""Концепция 57: Gradient Boosting (для регрессии, идея).

Учим последовательность 'слабых' моделей: каждая следующая предсказывает
остаток (ошибку) предыдущих. Итоговое предсказание = сумма.
"""
import numpy as np
from sklearn.tree import DecisionTreeRegressor

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 1))
y = X[:, 0] ** 2 + rng.normal(scale=0.2, size=200)

pred = np.zeros_like(y)
lr = 0.1
trees = []
for t in range(50):
    residual = y - pred
    tr = DecisionTreeRegressor(max_depth=2).fit(X, residual)
    pred += lr * tr.predict(X)
    trees.append(tr)

mse = ((y - pred) ** 2).mean()
print(f"Train MSE после 50 деревьев: {mse:.4f}")
