"""Раздел 56 — Оптимизация.

Файл 4: Методы второго порядка.

Концепция: Метод Ньютона.
x_{k+1} = x_k - H(x_k)^{-1} grad(x_k). На квадратичной f(x)=0.5*x^T A x - b^T x
один шаг сразу дает точное решение x*=A^{-1} b.

Концепция: Квази-Ньютон (BFGS).
Аппроксимация обратного гессиана B обновляется по парам (s,y), s=dx, y=dg:
B_{k+1} = (I - rho s y^T) B_k (I - rho y s^T) + rho s s^T, rho=1/(y^T s).
Сохраняет положительную определенность.

Концепция: L-BFGS.
"Limited-memory" BFGS — хранит только последние m пар (s,y) и применяет
B*v через two-loop рекурсию. Подходит для очень больших размерностей.

Концепция: Метод сопряженных градиентов (CG).
Решает Ax=b для симметричной положительно определенной A за <=n шагов.
Использует A-сопряженные направления: p_{k+1}=r_{k+1}+beta*p_k.
"""
import numpy as np

rng = np.random.default_rng(0)

# Квадратика
A = np.array([[3.0, 1.0], [1.0, 2.0]])
b = np.array([1.0, 2.0])
grad = lambda x: A @ x - b
H = A  # гессиан константный
x_star = np.linalg.solve(A, b)

# === Newton: один шаг ===
x = np.array([5.0, -3.0])
x_new = x - np.linalg.solve(H, grad(x))
print(f"Newton за 1 шаг: x={x_new.round(4)}, x*={x_star.round(4)}")

# === BFGS: ручное обновление за пару шагов ===
B = np.eye(2)  # начальная аппроксимация обратного гессиана
x = np.array([5.0, -3.0]); g = grad(x)
for _ in range(5):
    p = -B @ g
    x_new = x + p  # упрощенный шаг без line search
    g_new = grad(x_new)
    s = x_new - x
    yv = g_new - g
    rho = 1.0 / (yv @ s)
    I = np.eye(2)
    B = (I - rho * np.outer(s, yv)) @ B @ (I - rho * np.outer(yv, s)) + rho * np.outer(s, s)
    x, g = x_new, g_new
print(f"BFGS (5 шагов): x={x.round(4)}, ||g||={np.linalg.norm(g):.2e}")

# === L-BFGS: two-loop рекурсия для вычисления d=-B*g ===
hist_s, hist_y = [], []
def lbfgs_dir(g, hist_s, hist_y):
    q = g.copy(); alphas = []
    for s, yv in zip(reversed(hist_s), reversed(hist_y)):
        a = (s @ q) / (yv @ s); alphas.append(a); q = q - a * yv
    r = q  # H0 = I
    for (s, yv), a in zip(zip(hist_s, hist_y), reversed(alphas)):
        beta = (yv @ r) / (yv @ s); r = r + (a - beta) * s
    return -r

x = np.array([5.0, -3.0]); g = grad(x)
for _ in range(8):
    d = lbfgs_dir(g, hist_s, hist_y) if hist_s else -g
    x_new = x + 0.5 * d
    g_new = grad(x_new)
    hist_s.append(x_new - x); hist_y.append(g_new - g)
    if len(hist_s) > 3: hist_s.pop(0); hist_y.pop(0)
    x, g = x_new, g_new
print(f"L-BFGS (m=3): x={x.round(4)}")

# === CG: точное решение за <=n итераций ===
x = np.zeros(2); r = b - A @ x; p = r.copy()
for _ in range(2):
    Ap = A @ p
    alpha = (r @ r) / (p @ Ap)
    x = x + alpha * p
    r_new = r - alpha * Ap
    beta = (r_new @ r_new) / (r @ r)
    p = r_new + beta * p
    r = r_new
print(f"CG за 2 шага: x={x.round(6)}, x*={x_star.round(6)}")
