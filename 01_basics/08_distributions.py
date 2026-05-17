"""Концепция 8: Распределения - равномерное и нормальное.

Равномерное: все значения в диапазоне равновероятны.
Нормальное (гауссово): "колокол" - значения вокруг среднего.
"""
import numpy as np

rng = np.random.default_rng(42)

uni = rng.uniform(0, 10, size=10000)
norm = rng.normal(loc=5, scale=2, size=10000)

print("Равномерное [0,10]: среднее=%.2f, std=%.2f" % (uni.mean(), uni.std()))
print("Нормальное mu=5 sig=2: среднее=%.2f, std=%.2f" % (norm.mean(), norm.std()))
print("Около 68%% значений нормали в [mu-sig, mu+sig]:",
      ((norm > 3) & (norm < 7)).mean())
