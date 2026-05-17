"""Концепция 53: Information Gain.

Сколько 'неопределённости' мы убираем, разбивая множество по какому-то признаку.
IG = entropy(parent) - sum(|child|/|parent| * entropy(child)).
Дерево решений на каждом шаге выбирает split с максимальным IG.
"""
import numpy as np


def entropy(y):
    if len(y) == 0:
        return 0
    _, c = np.unique(y, return_counts=True)
    p = c / c.sum()
    return -np.sum(p * np.log2(p + 1e-12))


X = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

for thr in [2.5, 4.5, 6.5]:
    left = y[X <= thr]; right = y[X > thr]
    ig = entropy(y) - (len(left) / len(y) * entropy(left)
                       + len(right) / len(y) * entropy(right))
    print(f"thr={thr}: IG={ig:.3f}")
print("Лучший split там, где IG максимальный.")
