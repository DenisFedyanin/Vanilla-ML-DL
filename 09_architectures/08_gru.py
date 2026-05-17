"""Концепция 95: GRU - упрощённый LSTM с 2 гейтами (reset, update).

h_t = (1-z) * h_{t-1} + z * tilde_h.
Меньше параметров, часто работает не хуже LSTM.
"""
import numpy as np


def sigmoid(z): return 1 / (1 + np.exp(-z))


rng = np.random.default_rng(0)
in_dim, hid = 3, 4
Wz = rng.normal(size=(hid, in_dim + hid)) * 0.3
Wr = rng.normal(size=(hid, in_dim + hid)) * 0.3
Wh = rng.normal(size=(hid, in_dim + hid)) * 0.3

x = rng.normal(size=(5, in_dim))
h = np.zeros(hid)
for t in range(len(x)):
    xh = np.concatenate([x[t], h])
    z = sigmoid(Wz @ xh)
    r = sigmoid(Wr @ xh)
    htilde = np.tanh(Wh @ np.concatenate([x[t], r * h]))
    h = (1 - z) * h + z * htilde
print("Финальное h:", np.round(h, 3))
