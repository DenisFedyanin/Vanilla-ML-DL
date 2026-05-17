"""Концепция 28: Accuracy.

Доля правильных ответов. Плоха при сильном дисбалансе.
"""
import numpy as np

y = np.array([0, 1, 1, 0, 1, 1, 0])
p = np.array([0, 1, 0, 0, 1, 1, 1])

acc = (y == p).mean()
print("Accuracy =", acc)
