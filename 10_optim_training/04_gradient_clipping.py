"""Концепция 101: Gradient clipping.

Если ||grad|| > порога, масштабируем градиент так, чтобы норма стала равной порогу.
Спасает обучение RNN от взрывов.
"""
import numpy as np


def clip(grad, max_norm=1.0):
    norm = np.linalg.norm(grad)
    if norm > max_norm:
        return grad * (max_norm / norm)
    return grad


g_big = np.array([3.0, 4.0])
g_small = np.array([0.3, 0.4])
print("Норма большого:", np.linalg.norm(g_big), "-> после clip:",
      np.linalg.norm(clip(g_big, 1.0)))
print("Норма маленького:", np.linalg.norm(g_small), "-> после clip:",
      np.linalg.norm(clip(g_small, 1.0)))
