"""Раздел 56 — Оптимизация.

Файл 3: Координатные методы.

Концепция: Координатный спуск (CD).
По очереди минимизируем по одной координате, остальные фиксируем.
Для квадратичной f(x)=0.5*x^T A x - b^T x шаг по i: x_i* = (b_i - sum_{j!=i} A_ij x_j) / A_ii.

Концепция: Циклический CD.
Координаты перебираются в фиксированном порядке 1,2,...,n,1,2,... — самый простой вариант.

Концепция: Случайный CD.
На каждом шаге выбирается случайная координата. Часто сходится быстрее на практике.

Концепция: Блочный CD (BCD).
Координаты разбиваются на блоки; на шаге минимизируем по целому блоку.
Применяется когда блоковая задача решается аналитически (например, ALS в MF).
"""
import numpy as np

rng = np.random.default_rng(0)

# Задача наименьших квадратов: min ||X w - y||^2
n, d = 60, 5
X = rng.normal(size=(n, d))
w_true = np.array([1.0, -2.0, 0.5, 0.0, 3.0])
y = X @ w_true + 0.05 * rng.normal(size=n)

# Эквивалентно: A = X^T X, b = X^T y -> min 0.5 w^T A w - b^T w
A = X.T @ X
b = X.T @ y
w_star = np.linalg.solve(A, b)

# === Циклический CD ===
w = np.zeros(d)
for _ in range(50):
    for i in range(d):
        w[i] = (b[i] - A[i] @ w + A[i, i] * w[i]) / A[i, i]
print(f"Циклический CD: w={w.round(3)}, ||w-w*||={np.linalg.norm(w-w_star):.2e}")

# === Случайный CD ===
w = np.zeros(d)
for _ in range(200):
    i = rng.integers(d)
    w[i] = (b[i] - A[i] @ w + A[i, i] * w[i]) / A[i, i]
print(f"Случайный CD: w={w.round(3)}, ||w-w*||={np.linalg.norm(w-w_star):.2e}")

# === Блочный CD: блоки [0,1] и [2,3,4] ===
# w_B* = A_BB^{-1} (b_B - A_B,~B w_~B)
blocks = [[0, 1], [2, 3, 4]]
w = np.zeros(d)
for _ in range(20):
    for B in blocks:
        notB = [j for j in range(d) if j not in B]
        rhs = b[B] - A[np.ix_(B, notB)] @ w[notB]
        w[B] = np.linalg.solve(A[np.ix_(B, B)], rhs)
print(f"Блочный CD: w={w.round(3)}, ||w-w*||={np.linalg.norm(w-w_star):.2e}")
