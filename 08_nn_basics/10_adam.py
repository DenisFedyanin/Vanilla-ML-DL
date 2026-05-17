"""Концепция 86: Adam.

Комбинирует momentum (1й момент) и адаптивный lr (2й момент - RMSProp-стиль).
Самый популярный оптимизатор по умолчанию.
"""
import numpy as np

w = np.array([3.0, -4.0])
m = np.zeros_like(w); v = np.zeros_like(w)
b1, b2, eps, lr = 0.9, 0.999, 1e-8, 0.1

def grad(w): return np.array([2 * w[0], 6 * w[1]])

for t in range(1, 50):
    g = grad(w)
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g ** 2
    m_hat = m / (1 - b1 ** t)
    v_hat = v / (1 - b2 ** t)
    w = w - lr * m_hat / (np.sqrt(v_hat) + eps)

print("Минимум по Adam:", w)
