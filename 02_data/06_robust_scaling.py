"""Концепция 18: Robust scaling (по медиане и IQR).

Устойчив к выбросам в отличие от z-score.
x' = (x - median) / IQR, где IQR = Q75 - Q25.
"""
import numpy as np

X = np.array([1.0, 2, 3, 4, 5, 1000])  # 1000 - выброс

med = np.median(X)
q25, q75 = np.percentile(X, [25, 75])
iqr = q75 - q25

X_robust = (X - med) / iqr
X_zscore = (X - X.mean()) / X.std()

print("Robust:", X_robust)
print("Z-score:", X_zscore)
print("Робастное масштабирование меньше 'портится' выбросом 1000.")
