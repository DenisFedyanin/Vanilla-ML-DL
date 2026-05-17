"""Концепция 32: ROC-кривая и AUC.

ROC показывает trade-off между True Positive Rate и False Positive Rate
при разных порогах. AUC = площадь под кривой. 1.0 = идеал, 0.5 = случайно.
"""
import numpy as np

rng = np.random.default_rng(0)
n = 200
scores = np.concatenate([rng.normal(0.7, 0.2, n), rng.normal(0.3, 0.2, n)])
y = np.concatenate([np.ones(n), np.zeros(n)])

order = np.argsort(-scores)
y_sorted = y[order]

# Эффективная формула AUC через число "правильных" пар:
pos = np.sum(y == 1)
neg = np.sum(y == 0)
ranks = np.empty_like(order, dtype=float)
ranks[order] = np.arange(1, len(order) + 1)
auc = (np.sum(ranks[y == 1]) - pos * (pos + 1) / 2) / (pos * neg)
print(f"AUC = {auc:.3f}")
