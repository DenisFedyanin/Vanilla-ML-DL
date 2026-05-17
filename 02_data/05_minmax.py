"""Концепция 17: Min-Max нормализация.

Загоняем значения в отрезок [0, 1]: x' = (x - min) / (max - min).
"""
import numpy as np

X = np.array([3.0, 7, 1, 9, 5])
Xn = (X - X.min()) / (X.max() - X.min())
print("Исходные:", X)
print("В [0,1] :", Xn)
