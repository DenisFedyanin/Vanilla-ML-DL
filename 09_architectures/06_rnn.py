"""Концепция 93: RNN-ячейка - вручную.

h_t = tanh(W_x x_t + W_h h_{t-1} + b). Так сеть 'помнит' прошлое.
"""
import numpy as np

rng = np.random.default_rng(0)
T = 5; in_dim = 3; hid = 4

Wx = rng.normal(size=(hid, in_dim)) * 0.3
Wh = rng.normal(size=(hid, hid)) * 0.3
b = np.zeros(hid)

x_seq = rng.normal(size=(T, in_dim))
h = np.zeros(hid)
for t in range(T):
    h = np.tanh(Wx @ x_seq[t] + Wh @ h + b)
    print(f"шаг {t}: h = {np.round(h, 3)}")
