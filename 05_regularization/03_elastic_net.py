"""Концепция 46: Elastic Net = L1 + L2.

Сочетает плюсы Lasso (отбор признаков) и Ridge (устойчивость).
"""
import numpy as np
from sklearn.linear_model import ElasticNet

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 5))
y = 2 * X[:, 0] + X[:, 1] + rng.normal(scale=0.1, size=100)

m = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X, y)
print("Elastic Net coefs:", np.round(m.coef_, 3))
