"""Концепция 22: Обнаружение выбросов.

Два классических метода: правило 3 sigma и IQR (межквартильный размах).
"""
import numpy as np

X = np.array([10, 11, 12, 13, 14, 15, 100])

# Z-score
z = (X - X.mean()) / X.std()
outliers_z = X[np.abs(z) > 2]

# IQR
q1, q3 = np.percentile(X, [25, 75])
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outliers_iqr = X[(X < lo) | (X > hi)]

print("Выбросы по z-score :", outliers_z)
print("Выбросы по IQR     :", outliers_iqr)
