"""Концепция 94: LSTM ячейка (концептуально).

В LSTM 4 'гейта': forget, input, candidate, output.
c_t = f * c_{t-1} + i * g;   h_t = o * tanh(c_t).
Решает проблему vanishing gradient в длинных последовательностях.
"""
import numpy as np


def sigmoid(z): return 1 / (1 + np.exp(-z))


rng = np.random.default_rng(0)
in_dim, hid = 3, 4
# объединённая матрица для 4 гейтов
W = rng.normal(size=(4 * hid, in_dim + hid)) * 0.3
b = np.zeros(4 * hid)

x = rng.normal(size=(5, in_dim))
h = np.zeros(hid); c = np.zeros(hid)
for t in range(len(x)):
    z = W @ np.concatenate([x[t], h]) + b
    i, f, g, o = np.split(z, 4)
    i, f, o = sigmoid(i), sigmoid(f), sigmoid(o)
    g = np.tanh(g)
    c = f * c + i * g
    h = o * np.tanh(c)
print("Финальное h:", np.round(h, 3))
print("Финальное c:", np.round(c, 3))
