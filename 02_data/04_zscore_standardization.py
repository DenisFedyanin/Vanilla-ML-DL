"""Концепция 16: Z-score стандартизация.

x_new = (x - mean) / std. После этого среднее=0, std=1.
Помогает многим алгоритмам (особенно градиентным).
"""
import numpy as np

X = np.array([10.0, 20, 30, 40, 50])
mu, sigma = X.mean(), X.std()
Z = (X - mu) / sigma

print("X     :", X)
print("Z     :", Z)
print("mean  :", Z.mean(), " std :", Z.std())
