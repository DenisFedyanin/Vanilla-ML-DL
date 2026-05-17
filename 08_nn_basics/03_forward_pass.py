"""Концепция 78: Forward pass (прямой проход).

Просто последовательно применяем слои: a0 = x, a_{l+1} = act(W_l a_l + b_l).
"""
import numpy as np

rng = np.random.default_rng(0)
x = rng.normal(size=4)

W1 = rng.normal(size=(5, 4)); b1 = np.zeros(5)
W2 = rng.normal(size=(3, 5)); b2 = np.zeros(3)


def relu(z): return np.maximum(0, z)
def softmax(z):
    e = np.exp(z - z.max()); return e / e.sum()


h = relu(W1 @ x + b1)
y = softmax(W2 @ h + b2)
print("Вход:", x)
print("Скрытый слой:", h)
print("Выход (вероятности):", y)
