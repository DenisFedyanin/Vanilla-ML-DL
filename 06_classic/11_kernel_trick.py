"""Концепция 59: Kernel trick.

Идея: чтобы линейный алгоритм работал в нелинейном пространстве,
нам нужны только СКАЛЯРНЫЕ ПРОИЗВЕДЕНИЯ между точками.
Ядро K(x, x') заменяет это произведение, не делая отображения явно.

RBF: K(x,x') = exp(-gamma * ||x-x'||^2).
"""
import numpy as np


def rbf(x1, x2, gamma=0.5):
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)


a, b = np.array([1.0, 0.0]), np.array([0.5, 0.5])
print("RBF(a, a) =", rbf(a, a))
print("RBF(a, b) =", rbf(a, b))
print("RBF(a, b*10) =", rbf(a, b * 10))
print("Чем дальше точки, тем меньше значение ядра.")
