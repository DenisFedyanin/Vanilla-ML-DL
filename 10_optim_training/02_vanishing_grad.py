"""Концепция 99: Vanishing gradient.

В глубоких сетях с sigmoid градиент через много слоёв 'тает':
производная sigmoid максимум 0.25, и при 10 слоях имеем 0.25^10 ~ 1e-6.
"""
import numpy as np

# Производная sigmoid в нуле = 0.25.
factor = 0.25
for layers in [1, 5, 10, 20, 50]:
    print(f"После {layers:>2} слоёв градиент уменьшается в ~{factor ** layers:.2e} раз")
