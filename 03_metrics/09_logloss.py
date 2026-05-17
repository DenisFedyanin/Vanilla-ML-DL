"""Концепция 33: Log-loss (бинарная cross-entropy).

L = -mean( y*log(p) + (1-y)*log(1-p) )
Штрафует уверенно неправильные предсказания очень сильно.
"""
import numpy as np

y = np.array([1, 0, 1, 1, 0])
p = np.array([0.9, 0.1, 0.8, 0.4, 0.3])
eps = 1e-12
p = np.clip(p, eps, 1 - eps)

logloss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
print("Log-loss =", logloss)
