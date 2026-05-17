"""Концепции 52: Энтропия и Gini impurity.

Меры 'смешанности' классов в узле дерева.
Энтропия: -sum p_i log p_i.   Gini: 1 - sum p_i^2.
"""
import numpy as np


def entropy(y):
    _, c = np.unique(y, return_counts=True)
    p = c / c.sum()
    return -np.sum(p * np.log2(p + 1e-12))


def gini(y):
    _, c = np.unique(y, return_counts=True)
    p = c / c.sum()
    return 1 - np.sum(p ** 2)


for y in [[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 2, 2]]:
    y = np.array(y)
    print(f"y={y}  entropy={entropy(y):.3f}  gini={gini(y):.3f}")
