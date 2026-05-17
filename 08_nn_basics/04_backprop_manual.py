"""Концепция 79: Обратное распространение ошибки (backprop), руками.

Маленькая сеть 1 -> 4(relu) -> 1, учится y = x^2 на [-2,2].
Все градиенты выведены вручную из правила цепи.
"""
import numpy as np

rng = np.random.default_rng(0)
X = np.linspace(-2, 2, 100).reshape(-1, 1)
y = X ** 2

W1 = rng.normal(size=(1, 4)) * 0.5; b1 = np.zeros((1, 4))
W2 = rng.normal(size=(4, 1)) * 0.5; b2 = np.zeros((1, 1))

lr = 0.01
for epoch in range(2000):
    Z1 = X @ W1 + b1
    A1 = np.maximum(0, Z1)             # relu
    Yh = A1 @ W2 + b2

    loss = ((Yh - y) ** 2).mean()
    if epoch % 500 == 0:
        print(f"epoch {epoch:>4}: loss={loss:.4f}")

    dYh = 2 * (Yh - y) / len(X)
    dW2 = A1.T @ dYh
    db2 = dYh.sum(0, keepdims=True)
    dA1 = dYh @ W2.T
    dZ1 = dA1 * (Z1 > 0)
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(0, keepdims=True)

    W1 -= lr * dW1; b1 -= lr * db1
    W2 -= lr * dW2; b2 -= lr * db2

print("Финальный loss:", loss)
