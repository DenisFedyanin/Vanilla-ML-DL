"""Концепция 48: Dropout - концепция.

Во время обучения случайно 'выключаем' часть нейронов с вероятностью p.
Это заставляет сеть не полагаться на отдельные нейроны.
"""
import numpy as np

rng = np.random.default_rng(0)
activations = rng.normal(size=10)
p = 0.5
mask = (rng.uniform(size=10) > p).astype(float)
dropped = activations * mask / (1 - p)   # inverse scaling

print("Активации до:    ", np.round(activations, 2))
print("Маска (0=выкл):  ", mask.astype(int))
print("Активации после: ", np.round(dropped, 2))
