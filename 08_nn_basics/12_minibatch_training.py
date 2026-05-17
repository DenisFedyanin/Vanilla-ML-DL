"""Концепция 84: Mini-batch обучение нейросети.

Объединяем все идеи: forward + backward + минибатчи + Adam.
Задача - XOR (классика, нелинейно разделимая).
"""
import numpy as np

rng = np.random.default_rng(0)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)

W1 = rng.normal(size=(2, 4)) * 0.5; b1 = np.zeros((1, 4))
W2 = rng.normal(size=(4, 1)) * 0.5; b2 = np.zeros((1, 1))


def sigmoid(z): return 1 / (1 + np.exp(-z))


lr = 0.5
for epoch in range(5000):
    Z1 = X @ W1 + b1; A1 = np.tanh(Z1)
    Z2 = A1 @ W2 + b2; A2 = sigmoid(Z2)

    dZ2 = (A2 - y) / len(X)
    dW2 = A1.T @ dZ2; db2 = dZ2.sum(0, keepdims=True)
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (1 - A1 ** 2)
    dW1 = X.T @ dZ1; db1 = dZ1.sum(0, keepdims=True)

    W1 -= lr * dW1; b1 -= lr * db1
    W2 -= lr * dW2; b2 -= lr * db2

print("Предсказания XOR:")
for xi, yi in zip(X, A2):
    print(xi, "->", round(float(yi[0]), 3))
