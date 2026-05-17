"""Концепция 1: Векторы, матрицы, тензоры.

Тензор - это просто массив с произвольным числом осей.
"""
import numpy as np

scalar = np.array(7)              # 0D тензор
vector = np.array([1, 2, 3])      # 1D тензор (вектор)
matrix = np.array([[1, 2], [3, 4]])  # 2D тензор (матрица)
tensor3d = np.random.rand(2, 3, 4)   # 3D тензор

for name, t in [("scalar", scalar), ("vector", vector),
                ("matrix", matrix), ("tensor3d", tensor3d)]:
    print(f"{name}: shape={t.shape}, ndim={t.ndim}, dtype={t.dtype}")
