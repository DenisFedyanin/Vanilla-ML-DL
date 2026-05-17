"""Концепция 5: Численная производная и градиент.

Производная = насколько функция меняется при маленьком сдвиге аргумента.
Градиент = вектор частных производных. Указывает в сторону самого быстрого роста.
"""
import numpy as np


def f(x):
    return x[0] ** 2 + 3 * x[1] ** 2  # минимум в (0,0)


def num_grad(f, x, eps=1e-5):
    g = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        e = np.zeros_like(x, dtype=float); e[i] = eps
        g[i] = (f(x + e) - f(x - e)) / (2 * eps)
    return g


x = np.array([2.0, -1.0])
print("Точка:", x, "f(x)=", f(x))
print("Численный градиент:", num_grad(f, x))
print("Аналитический grad: [2*x0, 6*x1] =", [2 * x[0], 6 * x[1]])
