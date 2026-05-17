"""Концепция 27: Коэффициент детерминации R^2.

R^2 = 1 - SS_res / SS_tot.
1.0 = идеальная модель, 0 = не лучше среднего, <0 = хуже среднего.
"""
import numpy as np

y_true = np.array([3.0, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - y_true.mean()) ** 2)
r2 = 1 - ss_res / ss_tot
print("R^2 =", r2)
