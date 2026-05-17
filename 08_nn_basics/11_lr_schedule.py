"""Концепция 87: Learning rate schedule.

Большой lr в начале (быстрое обучение), потом уменьшаем (точная подстройка).
Step, exponential, cosine - популярные варианты.
"""
import numpy as np

epochs = 20
lr0 = 1.0

step = [lr0 * 0.5 ** (e // 5) for e in range(epochs)]
expo = [lr0 * 0.9 ** e for e in range(epochs)]
cos = [0.5 * lr0 * (1 + np.cos(np.pi * e / epochs)) for e in range(epochs)]

for e in range(epochs):
    print(f"ep{e:>2}  step={step[e]:.4f}  exp={expo[e]:.4f}  cos={cos[e]:.4f}")
