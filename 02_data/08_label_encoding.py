"""Концепция 20: Label encoding.

Просто заменяем категории числами 0, 1, 2, ...
Внимание: для линейных моделей это вводит ложный порядок!
Для деревьев - часто ок.
"""
import numpy as np

labels = np.array(["A", "B", "A", "C", "B", "A"])
classes, encoded = np.unique(labels, return_inverse=True)
print("Классы :", classes)
print("Коды   :", encoded)
