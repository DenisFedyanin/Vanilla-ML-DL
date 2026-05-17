"""Концепция 72: Один нейрон и линейный слой.

Нейрон: y = activation(w*x + b). Слой - много нейронов = матрица W.
"""
import numpy as np

x = np.array([1.0, 2.0, 3.0])
w = np.array([0.2, -0.3, 0.5])
b = 0.1
y = w @ x + b
print(f"Выход нейрона: {y}")

# Линейный слой: 3 -> 4 нейронов
W = np.random.default_rng(0).normal(size=(4, 3))
b = np.zeros(4)
out = W @ x + b
print("Выход линейного слоя (4 нейрона):", out)
