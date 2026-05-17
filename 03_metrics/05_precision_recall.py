"""Концепция 29: Precision и Recall.

Precision = TP / (TP + FP)  - "из тех, что я назвал положительными, сколько действительно?"
Recall    = TP / (TP + FN)  - "из всех настоящих положительных, сколько я нашёл?"
"""
import numpy as np

y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
p = np.array([1, 1, 0, 0, 1, 0, 0, 0])

TP = ((y == 1) & (p == 1)).sum()
FP = ((y == 0) & (p == 1)).sum()
FN = ((y == 1) & (p == 0)).sum()

precision = TP / (TP + FP)
recall = TP / (TP + FN)
print(f"TP={TP} FP={FP} FN={FN}")
print(f"Precision = {precision:.3f}")
print(f"Recall    = {recall:.3f}")
