"""Концепция 81: He инициализация.

Веса ~ N(0, 2 / fan_in). Хорошо подходит для ReLU
(половина активаций - нули, поэтому коэффициент 2).
"""
import numpy as np

rng = np.random.default_rng(0)
fan_in, fan_out = 100, 50
W = rng.normal(0, np.sqrt(2.0 / fan_in), size=(fan_out, fan_in))

x = rng.normal(size=fan_in)
out = np.maximum(0, W @ x)
print("std до ReLU: %.3f, std после ReLU: %.3f" % ((W @ x).std(), out.std()))
