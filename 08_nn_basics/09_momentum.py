"""Концепция 85: SGD с momentum.

v = beta * v - lr * grad;  w = w + v.
'Скользит' по оврагам функции потерь быстрее обычного SGD.
"""
import numpy as np

w = np.array([3.0, 4.0])
v = np.zeros_like(w)

def grad(w): return np.array([2 * w[0], 6 * w[1]])  # минимум в (0,0)

for it in range(20):
    v = 0.9 * v - 0.1 * grad(w)
    w = w + v

print("Минимум найден около:", w)
