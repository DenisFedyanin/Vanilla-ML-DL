"""Концепция 26: MSE и RMSE.

MSE = среднее квадратов ошибок. Сильнее наказывает большие ошибки.
RMSE = sqrt(MSE) - в тех же единицах, что и целевая переменная.
"""
import numpy as np

y_true = np.array([3.0, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

mse = np.mean((y_true - y_pred) ** 2)
rmse = np.sqrt(mse)
print("MSE  =", mse)
print("RMSE =", rmse)
