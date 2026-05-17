"""Концепция 41: Сигмоида.

sigma(z) = 1 / (1 + exp(-z)).  Переводит любое число в [0, 1].
Производная: sigma'(z) = sigma(z) * (1 - sigma(z)).
"""
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


for z in [-3, -1, 0, 1, 3]:
    print(f"sigma({z:>2}) = {sigmoid(z):.3f}")
