"""Концепция 15: Стратифицированный split.

Сохраняем пропорции классов в train и test - важно при дисбалансе.
"""
import numpy as np

rng = np.random.default_rng(0)
y = np.array([0] * 90 + [1] * 10)
rng.shuffle(y)

# Обычный split может случайно отдать в test очень мало класса 1.
# Стратифицированный split берёт пропорционально из каждого класса.
train_idx, test_idx = [], []
for cls in np.unique(y):
    cls_idx = np.where(y == cls)[0]
    rng.shuffle(cls_idx)
    n_test = int(0.2 * len(cls_idx))
    test_idx.extend(cls_idx[:n_test])
    train_idx.extend(cls_idx[n_test:])

print("Доля класса 1 в исходных :", (y == 1).mean())
print("Доля класса 1 в train    :", (y[train_idx] == 1).mean())
print("Доля класса 1 в test     :", (y[test_idx] == 1).mean())
