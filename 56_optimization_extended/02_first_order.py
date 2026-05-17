"""Раздел 56 — Оптимизация.

Файл 2: Методы первого порядка.

Концепция: Градиентный спуск (GD).
x_{k+1} = x_k - eta * grad f(x_k). Самый базовый шаг "против градиента".
Для L-гладкой выпуклой f сходится со скоростью O(1/k).

Концепция: Метод Нестерова (NAG).
Использует "взгляд вперед": y_k = x_k + b*(x_k - x_{k-1}); x_{k+1}=y_k-eta*grad f(y_k).
Скорость O(1/k^2) — оптимальная для гладких выпуклых задач.

Концепция: Heavy ball (Polyak momentum).
x_{k+1} = x_k - eta*grad f(x_k) + b*(x_k - x_{k-1}).
Похож на NAG, но градиент берется в текущей точке.

Концепция: Субградиентный метод.
Для негладких выпуклых f используется любой субградиент g in d f(x).
Шаг x_{k+1}=x_k - eta_k*g, eta_k=O(1/sqrt(k)). Медленнее GD.

Концепция: Proximal оператор + soft-thresholding для L1.
prox_{lam||.||_1}(v) = sign(v)*max(|v|-lam, 0). Базовый шаг ISTA для Lasso.
"""
import numpy as np

rng = np.random.default_rng(0)

# Задача: f(x) = 0.5 * x^T A x - b^T x, A=diag(1,10) -> плохо обусловленная
A = np.diag([1.0, 10.0])
b = np.array([1.0, 1.0])
grad = lambda x: A @ x - b
fval = lambda x: 0.5 * x @ A @ x - b @ x
x_star = np.linalg.solve(A, b)

# === GD ===
x = np.zeros(2); eta = 0.18
for _ in range(50):
    x = x - eta * grad(x)
print(f"GD: x={x.round(3)}, gap={fval(x)-fval(x_star):.4e}")

# === Heavy ball ===
x = np.zeros(2); x_prev = x.copy(); beta = 0.5
for _ in range(50):
    x_new = x - eta * grad(x) + beta * (x - x_prev)
    x_prev, x = x, x_new
print(f"HeavyBall: x={x.round(3)}, gap={fval(x)-fval(x_star):.4e}")

# === Nesterov ===
x = np.zeros(2); x_prev = x.copy()
for _ in range(50):
    y = x + beta * (x - x_prev)
    x_new = y - eta * grad(y)
    x_prev, x = x, x_new
print(f"Nesterov: x={x.round(3)}, gap={fval(x)-fval(x_star):.4e}")

# === Субградиент: f(x) = |x| ===
xs = 2.0
for k in range(1, 200):
    g = np.sign(xs) if xs != 0 else 0.0
    xs = xs - (0.1 / np.sqrt(k)) * g
print(f"Субградиент |x|: x≈{xs:.4f} (минимум 0)")

# === Proximal: soft-thresholding ===
v = np.array([-2.0, -0.3, 0.0, 0.4, 1.5])
lam = 0.5
prox = np.sign(v) * np.maximum(np.abs(v) - lam, 0)
print(f"soft_threshold(v, {lam}) = {prox}")
