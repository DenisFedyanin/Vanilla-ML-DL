"""Концепция 44: L1 регуляризация (Lasso).

К потере добавляем lambda * sum|w|. Лишние веса становятся РОВНО нулём -
автоматический отбор признаков.
"""
import numpy as np
from sklearn.linear_model import Lasso

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 5))
y = 3 * X[:, 0] - 2 * X[:, 1] + rng.normal(scale=0.1, size=100)

for alpha in [0.01, 0.1, 0.5]:
    m = Lasso(alpha=alpha).fit(X, y)
    print(f"alpha={alpha}: coefs={np.round(m.coef_, 3)}")
print("Видно, что 'ненужные' признаки -> ровно 0.")
