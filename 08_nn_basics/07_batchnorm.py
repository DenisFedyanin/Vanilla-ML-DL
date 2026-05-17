"""Концепция 82: Batch Normalization.

Нормируем выходы слоя по батчу: y = gamma * (x - mu) / sigma + beta.
Стабилизирует обучение, позволяет использовать большие lr.
"""
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(loc=5, scale=3, size=(8, 4))

mu = X.mean(0); var = X.var(0)
gamma = np.ones(4); beta = np.zeros(4)
Y = gamma * (X - mu) / np.sqrt(var + 1e-5) + beta

print("До BN: mean=%.3f std=%.3f" % (X.mean(), X.std()))
print("После BN: mean=%.3f std=%.3f" % (Y.mean(), Y.std()))
