"""Раздел 68 — Kernel methods.

Файл 1: Обзор ядер.

Концепция: Kernel trick.
Ядро K(x, y) = <phi(x), phi(y)> — скалярное произведение в (возможно бесконечномерном)
признаковом пространстве. Алгоритмы, выражаемые через скалярные произведения
(SVM, ridge, PCA, k-means), можно 'кернелизовать' и работать в этом пространстве
неявно — не вычисляя phi.

Концепция: Linear kernel.
K(x, y) = x^T y. Базовый, эквивалентен обычной модели.

Концепция: Polynomial kernel.
K(x, y) = (x^T y + c)^d. Включает все полиномиальные взаимодействия степени до d.
Параметры: степень d, сдвиг c.

Концепция: RBF (Gaussian) kernel.
K(x, y) = exp(-gamma * ||x - y||^2). Бесконечномерный признак. gamma управляет
'шириной': большое gamma -> узкие пики (риск переобучения), малое -> почти константа.

Концепция: Laplace kernel.
K(x, y) = exp(-gamma * ||x - y||_1). Похож на RBF, но острее в нуле; устойчивее
к выбросам, чем RBF.

Концепция: Sigmoid kernel.
K(x, y) = tanh(alpha * x^T y + c). Не всегда положительно-полуопределённый;
исторически связан с нейросетями.

Концепция: String kernel (концепт).
Считает число общих подстрок (с весом) у двух строк. Используется в био- и текст-
классификации.

Концепция: Gram matrix.
K_ij = K(x_i, x_j) — n x n. Должна быть симметричной и PSD; на ней основаны все
кернелизованные методы.
"""
import numpy as np

rng = np.random.default_rng(0)

X = rng.normal(size=(5, 3))

def K_lin(X, Y):
    return X @ Y.T

def K_poly(X, Y, d=3, c=1):
    return (X @ Y.T + c) ** d

def K_rbf(X, Y, gamma=0.5):
    sq = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    return np.exp(-gamma * sq)

def K_lap(X, Y, gamma=0.5):
    l1 = np.abs(X[:, None, :] - Y[None, :, :]).sum(-1)
    return np.exp(-gamma * l1)

def K_sigm(X, Y, alpha=0.1, c=0):
    return np.tanh(alpha * X @ Y.T + c)

print("=== Gram-матрицы (5x5) ===")
np.set_printoptions(precision=2, suppress=True)
print("\nLinear:\n", K_lin(X, X))
print("\nPolynomial d=3:\n", K_poly(X, X))
print("\nRBF gamma=0.5:\n", K_rbf(X, X))
print("\nLaplace gamma=0.5:\n", K_lap(X, X))
print("\nSigmoid alpha=0.1:\n", K_sigm(X, X))

# === проверка PSD ===
for name, K in [("Linear", K_lin(X, X)),
                ("Poly", K_poly(X, X)),
                ("RBF", K_rbf(X, X)),
                ("Laplace", K_lap(X, X)),
                ("Sigmoid", K_sigm(X, X))]:
    ev = np.linalg.eigvalsh((K + K.T) / 2)
    print(f"{name:10s}: min eig = {ev.min():.4f}  ({'PSD' if ev.min() >= -1e-8 else 'NOT PSD'})")

# === gamma в RBF ===
print("\nЭффект gamma в RBF (значение K(x, x+0.5)):")
x = np.array([0.0, 0.0])
y = np.array([0.5, 0.0])
for g in [0.01, 0.1, 1.0, 10.0]:
    k = np.exp(-g * np.sum((x - y) ** 2))
    print(f"  gamma={g:5.2f} -> K={k:.4f}")

# === String kernel concept ===
def string_kernel(s, t, p=3):
    """Простейшее: число общих p-грамм."""
    sg = {s[i:i + p] for i in range(len(s) - p + 1)}
    tg = {t[i:i + p] for i in range(len(t) - p + 1)}
    return len(sg & tg)

print("\nString kernel (3-граммы):")
print(f"  K('machine', 'learning') = {string_kernel('machine', 'learning')}")
print(f"  K('learning', 'learning') = {string_kernel('learning', 'learning')}")
print(f"  K('teaching', 'learning') = {string_kernel('teaching', 'learning')}")
