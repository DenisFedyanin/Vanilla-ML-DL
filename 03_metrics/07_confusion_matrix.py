"""Концепция 31: Матрица ошибок (confusion matrix).

Строки - истинный класс, столбцы - предсказанный.
"""
import numpy as np

y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
p = np.array([0, 2, 2, 0, 0, 2, 0, 1, 1])
K = 3

cm = np.zeros((K, K), dtype=int)
for t, q in zip(y, p):
    cm[t, q] += 1

print("Confusion matrix:")
print(cm)
print("Диагональ = правильные предсказания, всё остальное = ошибки.")
