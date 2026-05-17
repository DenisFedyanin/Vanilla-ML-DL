"""Концепция 21: Заполнение пропусков.

Самые простые стратегии: средним, медианой, модой, константой.
"""
import numpy as np

X = np.array([1.0, 2.0, np.nan, 4.0, np.nan, 6.0])

X_mean = np.where(np.isnan(X), np.nanmean(X), X)
X_med = np.where(np.isnan(X), np.nanmedian(X), X)
X_zero = np.where(np.isnan(X), 0.0, X)

print("Исходные:", X)
print("Средним :", X_mean)
print("Медианой:", X_med)
print("Нулем   :", X_zero)
