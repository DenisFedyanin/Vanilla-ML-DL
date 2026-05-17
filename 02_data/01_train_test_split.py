"""Концепция 13: Train / Test split.

Делим данные на train (учим) и test (проверяем).
Никогда не подсматриваем в test!
"""
import numpy as np

rng = np.random.default_rng(0)
X = np.arange(20).reshape(-1, 1)
y = np.arange(20)

idx = rng.permutation(len(X))
test_size = 0.2
n_test = int(len(X) * test_size)
test_idx = idx[:n_test]
train_idx = idx[n_test:]

print("Train размер:", len(train_idx))
print("Test  размер:", len(test_idx))
print("train_idx :", train_idx)
print("test_idx  :", test_idx)
