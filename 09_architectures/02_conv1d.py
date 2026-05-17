"""Концепция 89: 1D-свёртка - вручную.

Скользящее окошко: на каждой позиции считаем скалярное произведение
с ядром (kernel).
"""
import numpy as np


def conv1d(x, k):
    out = np.zeros(len(x) - len(k) + 1)
    for i in range(len(out)):
        out[i] = np.dot(x[i:i + len(k)], k)
    return out


x = np.array([1.0, 2, 3, 4, 5, 6, 7])
k = np.array([1.0, 0, -1])    # ядро 'разность соседей' - детектор краёв
print("Сигнал:", x)
print("Ядро  :", k)
print("Conv  :", conv1d(x, k))
