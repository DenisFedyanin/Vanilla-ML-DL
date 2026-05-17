"""Концепция 96: Autoencoder.

Сеть учится восстанавливать вход после прохождения через 'узкое горлышко'.
encode: X -> z (низкая размерность), decode: z -> X_hat.
Loss = MSE(X, X_hat).
"""
import numpy as np

rng = np.random.default_rng(0)
N, D = 200, 8
# создадим 'настоящую' структуру в 2D, поднимем в 8D
true_z = rng.normal(size=(N, 2))
A = rng.normal(size=(2, D))
X = true_z @ A + 0.05 * rng.normal(size=(N, D))

We = rng.normal(size=(D, 2)) * 0.1
Wd = rng.normal(size=(2, D)) * 0.1

lr = 0.01
for ep in range(2000):
    Z = X @ We               # encode
    Xh = Z @ Wd              # decode
    err = Xh - X
    loss = (err ** 2).mean()
    dWd = Z.T @ err / N
    dZ = err @ Wd.T
    dWe = X.T @ dZ / N
    We -= lr * dWe; Wd -= lr * dWd

print(f"Финальный loss восстановления: {loss:.5f}")
print("Сжали 8D -> 2D и восстановили.")
