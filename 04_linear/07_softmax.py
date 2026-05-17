"""Концепция 42: Softmax.

Превращает вектор оценок в распределение вероятностей по классам.
softmax(z)_i = exp(z_i) / sum_j exp(z_j).
"""
import numpy as np


def softmax(z):
    z = z - z.max()                # для численной стабильности
    e = np.exp(z)
    return e / e.sum()


z = np.array([2.0, 1.0, 0.1])
p = softmax(z)
print("Оценки   :", z)
print("Вероятности:", p, " сумма =", p.sum())
