"""Раздел 30 — Оптимизаторы (расширенно). Файл 2: адаптивные.

Минимизируем f(x) = 0.5 * x^T A x с A = diag(10, 1, 0.1).

Концепция 326: Adagrad.
G_t = sum_{s<=t} g_s^2; x_{t+1} = x_t - lr * g_t / (sqrt(G_t) + eps).
Адаптирует шаг для каждой координаты. Минус: G растёт навсегда -> шаг -> 0.

Концепция 327: RMSProp.
v_t = rho * v_{t-1} + (1-rho) * g_t^2; x_{t+1} = x_t - lr * g / (sqrt(v) + eps).
Экспоненциальное скользящее среднее квадратов градиента — не "умирает".

Концепция 328: AdaDelta.
RMSProp без lr — масштаб шага считается из EMA квадратов самих шагов.
Делает оптимизатор почти без гиперпараметров.

Концепция 329: Adam = Momentum + RMSProp + bias correction.
m_t = b1*m + (1-b1)*g; v_t = b2*v + (1-b2)*g^2;
m_hat = m/(1-b1^t); v_hat = v/(1-b2^t); x -= lr * m_hat/(sqrt(v_hat)+eps).
Самый ходовой оптимизатор.

Концепция 330: AMSGrad.
Тот же Adam, но v_max = max(v_max, v) — гарантирует невозрастание шага.
Исправляет теоретический баг Adam.

Концепция 331: AdamW.
Adam с правильным weight decay: применяем wd*x ОТДЕЛЬНО от градиента (не как L2),
чтобы decay не нормировался на v. Стандарт для трансформеров.
"""
import numpy as np

A = np.diag([10.0, 1.0, 0.1])
x0 = np.array([1.0, 1.0, 1.0])
T, eps = 100, 1e-8
def f(x): return 0.5 * x @ A @ x
def grad(x): return A @ x

# === 326: Adagrad ===
x, G = x0.copy(), np.zeros(3)
for _ in range(T):
    g = grad(x); G += g * g
    x -= 0.5 * g / (np.sqrt(G) + eps)
print(f"326 Adagrad : f={f(x):.5f}, x={x.round(4).tolist()}")

# === 327: RMSProp ===
x, v = x0.copy(), np.zeros(3)
for _ in range(T):
    g = grad(x); v = 0.9 * v + 0.1 * g * g
    x -= 0.1 * g / (np.sqrt(v) + eps)
print(f"327 RMSProp : f={f(x):.5f}, x={x.round(4).tolist()}")

# === 328: AdaDelta ===
x, Eg, Edx = x0.copy(), np.zeros(3), np.zeros(3)
rho = 0.95
for _ in range(T):
    g = grad(x)
    Eg = rho * Eg + (1 - rho) * g * g
    dx = -np.sqrt(Edx + eps) / np.sqrt(Eg + eps) * g
    Edx = rho * Edx + (1 - rho) * dx * dx
    x = x + dx
print(f"328 AdaDelta: f={f(x):.5f}, x={x.round(4).tolist()} (без lr)")

# === 329: Adam ===
x, m, v = x0.copy(), np.zeros(3), np.zeros(3)
b1, b2, lr = 0.9, 0.999, 0.1
for t in range(1, T + 1):
    g = grad(x)
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    m_hat = m / (1 - b1 ** t)
    v_hat = v / (1 - b2 ** t)
    x -= lr * m_hat / (np.sqrt(v_hat) + eps)
print(f"329 Adam    : f={f(x):.5f}, x={x.round(4).tolist()}")

# === 330: AMSGrad ===
x, m, v, v_max = x0.copy(), np.zeros(3), np.zeros(3), np.zeros(3)
for t in range(1, T + 1):
    g = grad(x)
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    v_max = np.maximum(v_max, v)
    x -= lr * m / (np.sqrt(v_max) + eps)
print(f"330 AMSGrad : f={f(x):.5f}, x={x.round(4).tolist()}")

# === 331: AdamW ===
x, m, v = x0.copy(), np.zeros(3), np.zeros(3)
wd = 0.01
for t in range(1, T + 1):
    g = grad(x)
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    m_hat = m / (1 - b1 ** t); v_hat = v / (1 - b2 ** t)
    x -= lr * (m_hat / (np.sqrt(v_hat) + eps) + wd * x)
print(f"331 AdamW   : f={f(x):.5f}, x={x.round(4).tolist()}")
