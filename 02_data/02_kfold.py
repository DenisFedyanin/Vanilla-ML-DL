"""Концепция 14: K-Fold кросс-валидация.

Делим данные на K частей. Каждая часть по очереди становится тестом,
остальные - тренировкой. Это даёт более стабильную оценку качества.
"""
import numpy as np

n = 10
k = 5
idx = np.arange(n)
folds = np.array_split(idx, k)

for i in range(k):
    test = folds[i]
    train = np.concatenate([folds[j] for j in range(k) if j != i])
    print(f"Fold {i}: test={test.tolist()}  train={train.tolist()}")
