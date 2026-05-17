"""Раздел 60 — Анализ.

Файл 2: Продвинутые темы.

Концепция: Многомерное правило цепочки.
Для z = f(g(x)), где g: R^n->R^m, f: R^m->R:
dz/dx = (df/dg)^T (dg/dx) = J_f J_g. Лежит в основе backprop.

Концепция: Теорема о неявной функции (концепт).
Если F(x,y)=0, dF/dy != 0 в точке, то локально y = phi(x) и
dy/dx = -(dF/dy)^{-1} (dF/dx). Используется в bi-level optimization, meta-learning.

Концепция: Ряд Тейлора (1D и 2D).
1D: f(x+h) = f(x) + f'(x) h + f''(x) h^2/2 + ...
nD: f(x+dx) ≈ f(x) + grad^T dx + 0.5 dx^T H dx + ...
Используется для локального анализа и квадратичных моделей.

Концепция: Множители Лагранжа.
Для min f(x) s.t. h(x)=0: ищем стационарную точку L = f - lam*h:
grad f = lam grad h, h(x)=0. Дает явное решение для квадратичных задач с лин. ограничением.
"""
import numpy as np

# === Правило цепочки: z = sin(x^2 + y^2) ===
g = lambda x: np.array([x[0] ** 2 + x[1] ** 2])
f = lambda u: np.sin(u[0])
# dz/dx через chain rule
x = np.array([0.5, 0.3])
J_g = np.array([[2 * x[0], 2 * x[1]]])
J_f = np.array([[np.cos(g(x)[0])]])
dz_dx_chain = (J_f @ J_g).ravel()
# напрямую: z = sin(x^2+y^2), dz/dx_i = cos(x^2+y^2)*2 x_i
dz_dx_direct = np.cos(x[0] ** 2 + x[1] ** 2) * 2 * x
print(f"Chain rule: {dz_dx_chain.round(5)}, прямой: {dz_dx_direct.round(5)}")

# === Неявная функция: x^2 + y^2 - 1 = 0, dy/dx = -x/y ===
xp, yp = 0.6, np.sqrt(1 - 0.36)
dy_dx_impl = -xp / yp
# проверка численная
eps = 1e-4
yp_plus = np.sqrt(1 - (xp + eps) ** 2)
yp_minus = np.sqrt(1 - (xp - eps) ** 2)
print(f"Неявная: dy/dx теор={dy_dx_impl:.4f}, числ={(yp_plus-yp_minus)/(2*eps):.4f}")

# === Тейлор 1D: e^x вокруг 0 ===
def taylor_exp(x, n_terms=5):
    s = 0; t = 1.0
    for k in range(n_terms):
        if k > 0: t *= x / k
        s += t
    return s
print(f"exp(0.5) ≈ Тейлор(5)={taylor_exp(0.5):.5f}, true={np.exp(0.5):.5f}")

# === Тейлор 2D квадратичной аппроксимации f(x,y)=cos(x)cos(y) около (0,0) ===
f2 = lambda v: np.cos(v[0]) * np.cos(v[1])
# f(0,0)=1, grad=0, H = -I -> f ≈ 1 - 0.5*(x^2+y^2)
v = np.array([0.1, -0.2])
approx = 1 - 0.5 * (v[0] ** 2 + v[1] ** 2)
print(f"Тейлор 2D квадрат: {approx:.5f}, истинное: {f2(v):.5f}")

# === Лагранж: min x^2+y^2 s.t. x+y=1 -> x=y=0.5, lam=1 ===
# Решим как линейную систему: grad f = lam grad h, h=0
# 2x = lam, 2y = lam, x+y=1 -> x=y=lam/2 -> lam=1
A = np.array([[2.0, 0.0, -1.0], [0.0, 2.0, -1.0], [1.0, 1.0, 0.0]])
rhs = np.array([0.0, 0.0, 1.0])
sol = np.linalg.solve(A, rhs)
print(f"Lagrange: x={sol[0]:.4f}, y={sol[1]:.4f}, lambda={sol[2]:.4f}")
