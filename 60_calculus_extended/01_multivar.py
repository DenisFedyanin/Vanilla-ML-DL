"""Раздел 60 — Анализ (расширение).

Файл 1: Многомерное дифференцирование.

Концепция: Частные производные.
df/dx_i вычисляется как обычная производная, остальные переменные считаются константами.

Концепция: Градиент.
grad f(x) = (df/dx_1, ..., df/dx_n). Указывает направление наибыстрейшего роста.
Длина = скорость роста в этом направлении.

Концепция: Якобиан.
Для F: R^n -> R^m, J_{ij} = dF_i/dx_j — матрица m x n.
Лучшая локальная линейная аппроксимация. F(x+dx) ≈ F(x) + J dx.

Концепция: Гессиан.
Для f: R^n -> R, H_{ij} = d^2 f/(dx_i dx_j) — симметричная n x n.
Используется в Ньютоне, для классификации стационарных точек.

Концепция: Дивергенция и ротор (концепт).
Для векторного поля F: R^3->R^3, div F = sum dF_i/dx_i (скаляр),
curl F = (dF_3/dx_2 - dF_2/dx_3, ...) (вектор). Применяются в физике, PDE.
"""
import numpy as np

rng = np.random.default_rng(0)

# === Целевая f(x,y) = x^2 + 3 x y + 2 y^2 ===
f = lambda v: v[0] ** 2 + 3 * v[0] * v[1] + 2 * v[1] ** 2
# Аналитический градиент: (2x + 3y, 3x + 4y), Hessian = [[2,3],[3,4]]
grad_an = lambda v: np.array([2 * v[0] + 3 * v[1], 3 * v[0] + 4 * v[1]])
H_an = np.array([[2.0, 3.0], [3.0, 4.0]])

# === Численный градиент через центральные конечные разности ===
def num_grad(f, x, eps=1e-5):
    g = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x); e[i] = eps
        g[i] = (f(x + e) - f(x - e)) / (2 * eps)
    return g

x = np.array([0.5, -1.0])
print(f"Численный grad: {num_grad(f, x).round(5)}, аналитический: {grad_an(x).round(5)}")

# === Якобиан F: R^2 -> R^2 ===
F = lambda v: np.array([v[0] ** 2 - v[1], v[0] * v[1]])
def num_jac(F, x, eps=1e-5):
    m = len(F(x)); n = len(x)
    J = np.zeros((m, n))
    for j in range(n):
        e = np.zeros_like(x); e[j] = eps
        J[:, j] = (F(x + e) - F(x - e)) / (2 * eps)
    return J

J = num_jac(F, x)
# Аналитика: J = [[2x, -1], [y, x]]
J_an = np.array([[2 * x[0], -1.0], [x[1], x[0]]])
print(f"Численный Jac:\n{J.round(4)}\nАналитический:\n{J_an}")

# === Гессиан через двойные конечные разности ===
def num_hessian(f, x, eps=1e-4):
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ei = np.zeros_like(x); ei[i] = eps
            ej = np.zeros_like(x); ej[j] = eps
            H[i, j] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * eps * eps)
    return H

H_num = num_hessian(f, x)
print(f"Численный Hessian:\n{H_num.round(3)}\nАналитический:\n{H_an}")

# === Дивергенция и ротор поля F(x,y,z)=(yz, xz, xy) ===
# div = 0, curl = (x - x, y - y, z - z) = 0 -> поле бездивергентное и безвихревое
# Проверим численно
def F3(v): return np.array([v[1] * v[2], v[0] * v[2], v[0] * v[1]])
def div3(F, v, eps=1e-5):
    s = 0
    for i in range(3):
        e = np.zeros_like(v); e[i] = eps
        s += (F(v + e)[i] - F(v - e)[i]) / (2 * eps)
    return s
print(f"div F в точке [1,2,3] = {div3(F3, np.array([1.0,2.0,3.0])):.5f} (теория 0)")
