"""Концепция 35: MAPE - Mean Absolute Percentage Error.

Средний процент ошибки. Удобно интерпретировать, но не работает при y=0.
"""
import numpy as np

y = np.array([100.0, 200, 300, 400])
p = np.array([110.0, 190, 330, 380])

mape = np.mean(np.abs((y - p) / y)) * 100
print(f"MAPE = {mape:.2f} %")
