"""Концепция 25: MAE - Mean Absolute Error.

Среднее модулей ошибок. Не "наказывает" большие ошибки сильно.
"""
import numpy as np

y_true = np.array([3.0, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

mae = np.mean(np.abs(y_true - y_pred))
print("MAE =", mae)
