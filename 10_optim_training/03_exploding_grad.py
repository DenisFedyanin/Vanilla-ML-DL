"""Концепция 100: Exploding gradient.

Обратная проблема: если веса > 1, градиент 'взрывается' в глубоких сетях / RNN.
"""
import numpy as np

w_norm = 1.5
for steps in [1, 5, 10, 20, 50]:
    print(f"После {steps:>2} шагов: коэффициент роста ~ {w_norm ** steps:.2e}")
