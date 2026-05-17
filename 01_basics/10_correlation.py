"""Концепция 10: Корреляция Пирсона.

Показывает линейную связь между двумя величинами: -1 .. +1.
"""
import numpy as np

rng = np.random.default_rng(0)
x = rng.normal(size=200)
y_pos = 0.8 * x + 0.2 * rng.normal(size=200)
y_neg = -0.5 * x + 0.5 * rng.normal(size=200)
y_no = rng.normal(size=200)


def pearson(a, b):
    a_c = a - a.mean()
    b_c = b - b.mean()
    return (a_c * b_c).sum() / np.sqrt((a_c ** 2).sum() * (b_c ** 2).sum())


print("Положительная связь :", pearson(x, y_pos))
print("Отрицательная связь :", pearson(x, y_neg))
print("Нет связи           :", pearson(x, y_no))
