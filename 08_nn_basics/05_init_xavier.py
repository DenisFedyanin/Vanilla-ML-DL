"""Концепция 80: Xavier (Glorot) инициализация.

Веса ~ N(0, 1 / fan_in). Хорошо подходит для tanh/sigmoid.
Сохраняет дисперсию активаций через слои.
"""
import numpy as np

rng = np.random.default_rng(0)
fan_in, fan_out = 100, 50
W = rng.normal(0, np.sqrt(1.0 / fan_in), size=(fan_out, fan_in))

x = rng.normal(size=fan_in)
print("std входа: %.3f, std выхода: %.3f" % (x.std(), (W @ x).std()))
