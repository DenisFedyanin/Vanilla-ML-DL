"""Концепция 30: F1-score.

Среднее гармоническое precision и recall.
F1 = 2 * P * R / (P + R).
"""
import numpy as np

y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
p = np.array([1, 1, 0, 0, 1, 0, 0, 0])

TP = ((y == 1) & (p == 1)).sum()
FP = ((y == 0) & (p == 1)).sum()
FN = ((y == 1) & (p == 0)).sum()
prec = TP / (TP + FP)
rec = TP / (TP + FN)
f1 = 2 * prec * rec / (prec + rec)
print(f"F1 = {f1:.3f}")
