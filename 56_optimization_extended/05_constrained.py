"""Раздел 56 — Оптимизация.

Файл 5: Условная оптимизация.

Концепция: Лагранжиан.
Для задачи min f(x) s.t. g_i(x)<=0, h_j(x)=0:
L(x, lam, mu) = f(x) + sum lam_i g_i + sum mu_j h_j. Двойственная функция d(lam,mu)=inf_x L.

Концепция: Условия KKT.
В точке оптимума (выпуклый случай): stationarity grad_x L=0, primal feasibility,
dual feasibility lam>=0, complementary slackness lam_i g_i=0.

Концепция: Проектированный градиент (projected GD).
x_{k+1} = P_C(x_k - eta * grad f(x_k)). Для коробочного ограничения
P_C(x)=clip(x, lo, hi). Применяется в box-constrained QP, NMF и др.

Концепция: ADMM.
Разбивает задачу min f(x)+g(z) s.t. Ax+Bz=c на чередующиеся шаги по x, z и
обновление двойственной переменной. Для Lasso: x-шаг — ridge, z-шаг — soft-threshold.

Концепция: FISTA.
Ускоренная версия ISTA с инерцией Нестерова. Дает скорость O(1/k^2) для Lasso.
"""
import numpy as np

rng = np.random.default_rng(0)

# === Проектированный градиент на коробку ===
# min 0.5 ||x - c||^2 s.t. 0<=x<=1
c = np.array([1.5, -0.3, 0.7])
x = np.zeros(3)
for _ in range(30):
    x = np.clip(x - 0.5 * (x - c), 0.0, 1.0)
print(f"Box-projected GD: x={x} (ожидаемое clip(c,0,1)={np.clip(c,0,1)})")

# === KKT-проверка на простой задаче min x^2 s.t. x>=1 ===
# Лагранжиан: L=x^2 - lam*(x-1), grad=2x-lam=0, lam*(x-1)=0, lam>=0
# Решение x=1, lam=2.
x_kkt, lam = 1.0, 2.0
stationarity = 2 * x_kkt - lam
print(f"KKT: stat={stationarity}, complementarity={lam*(x_kkt-1)}, dual lam>=0: {lam>=0}")

# Lasso-задача: min 0.5||Xw-y||^2 + lam||w||_1
n, d = 50, 10
X = rng.normal(size=(n, d))
w_true = np.zeros(d); w_true[:3] = [2.0, -1.5, 1.0]
y = X @ w_true + 0.1 * rng.normal(size=n)
lam_l1 = 0.5

soft = lambda v, t: np.sign(v) * np.maximum(np.abs(v) - t, 0)
L_const = np.linalg.eigvalsh(X.T @ X).max()  # шаг 1/L

# === ISTA ===
w = np.zeros(d)
for _ in range(200):
    w = soft(w - (1 / L_const) * X.T @ (X @ w - y), lam_l1 / L_const)
print(f"ISTA: nnz={np.sum(np.abs(w)>1e-3)}, w[:3]={w[:3].round(3)}")

# === FISTA ===
w = np.zeros(d); z = w.copy(); t = 1.0
for _ in range(200):
    w_new = soft(z - (1 / L_const) * X.T @ (X @ z - y), lam_l1 / L_const)
    t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
    z = w_new + ((t - 1) / t_new) * (w_new - w)
    w, t = w_new, t_new
print(f"FISTA: nnz={np.sum(np.abs(w)>1e-3)}, w[:3]={w[:3].round(3)}")

# === ADMM для Lasso (концептуальный мини-цикл) ===
# min 0.5||Xw-y||^2 + lam||z||_1 s.t. w=z
rho = 1.0
w = np.zeros(d); z = np.zeros(d); u = np.zeros(d)
XtX_rho_inv = np.linalg.inv(X.T @ X + rho * np.eye(d))
Xty = X.T @ y
for _ in range(50):
    w = XtX_rho_inv @ (Xty + rho * (z - u))
    z = soft(w + u, lam_l1 / rho)
    u = u + (w - z)
print(f"ADMM Lasso: nnz={np.sum(np.abs(z)>1e-3)}, z[:3]={z[:3].round(3)}")
