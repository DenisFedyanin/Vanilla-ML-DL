"""Раздел 30 — Оптимизаторы (расширенно). Файл 3: современные.

Все методы делают один шаг на квадратичной функции f(x)=0.5 x^T A x.

Концепция 332: AdaMax.
Вместо v=EMA(g^2) используем u = max(b2*u, |g|) — L_inf-норма.
Стабильнее на разреженных градиентах.

Концепция 333: Nadam.
Adam + Nesterov: момент применяется с lookahead перед делением на sqrt(v).

Концепция 334: RAdam.
Rectified Adam: первые N шагов работает как SGD-with-momentum,
пока дисперсия EMA не "прогреется". Затем включает Adam-знаменатель.

Концепция 335: Lion (EvoLved Sign Momentum).
Использует только знак момента: update = sign(b1*m + (1-b1)*g).
Очень дешёвый, конкурирует с AdamW для трансформеров.

Концепция 336: LAMB.
Layer-wise Adam с нормированием обновления на ||w||/||update||.
Делает обучение возможным с очень большими батчами (BERT).

Концепция 337: LARS.
Layer-wise Adaptive Rate Scaling: lr_layer = lr * ||w|| / (||g|| + wd*||w||).
Аналогичная идея для SGD, ImageNet с большими батчами.
"""
import numpy as np

A = np.diag([5.0, 1.0])
x0 = np.array([1.0, 1.0])
def grad(x): return A @ x
def f(x): return 0.5 * x @ A @ x

eps, b1, b2 = 1e-8, 0.9, 0.999

# === 332: AdaMax (один шаг) ===
x, m, u = x0.copy(), np.zeros(2), np.zeros(2)
g = grad(x); m = b1 * m + (1 - b1) * g; u = np.maximum(b2 * u, np.abs(g))
x -= (0.1 / (1 - b1)) * m / (u + eps)
print(f"332 AdaMax step: x={x.round(4).tolist()}, f={f(x):.4f}")

# === 333: Nadam (один шаг) ===
x, m, v = x0.copy(), np.zeros(2), np.zeros(2)
g = grad(x); m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
m_hat = m / (1 - b1); v_hat = v / (1 - b2)
nesterov_m = b1 * m_hat + (1 - b1) * g / (1 - b1)
x -= 0.1 * nesterov_m / (np.sqrt(v_hat) + eps)
print(f"333 Nadam  step: x={x.round(4).tolist()}, f={f(x):.4f}")

# === 334: RAdam (один шаг, в "не прогретом" режиме = SGD+m) ===
x, m, v = x0.copy(), np.zeros(2), np.zeros(2)
t = 1
g = grad(x); m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
rho_inf = 2 / (1 - b2) - 1
rho_t = rho_inf - 2 * t * b2 ** t / (1 - b2 ** t)
m_hat = m / (1 - b1 ** t)
if rho_t > 4:
    v_hat = v / (1 - b2 ** t)
    r = np.sqrt((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t))
    x -= 0.1 * r * m_hat / (np.sqrt(v_hat) + eps)
else:
    x -= 0.1 * m_hat                  # SGD-mode на старте
print(f"334 RAdam  step: x={x.round(4).tolist()}, f={f(x):.4f} (rho_t={rho_t:.2f})")

# === 335: Lion ===
x, m = x0.copy(), np.zeros(2)
g = grad(x)
update = np.sign(b1 * m + (1 - b1) * g)
x -= 0.1 * update
m = b1 * m + (1 - b1) * g
print(f"335 Lion   step: x={x.round(4).tolist()}, f={f(x):.4f}")

# === 336: LAMB ===
x, m, v = x0.copy(), np.zeros(2), np.zeros(2)
g = grad(x); m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
m_hat = m / (1 - b1); v_hat = v / (1 - b2)
update = m_hat / (np.sqrt(v_hat) + eps)
ratio = np.linalg.norm(x) / (np.linalg.norm(update) + eps)
x -= 0.1 * ratio * update
print(f"336 LAMB   step: x={x.round(4).tolist()}, f={f(x):.4f}, layer-ratio={ratio:.3f}")

# === 337: LARS ===
x, v = x0.copy(), np.zeros(2)
g = grad(x); wd = 0.01
local_lr = np.linalg.norm(x) / (np.linalg.norm(g) + wd * np.linalg.norm(x) + eps)
v = 0.9 * v + local_lr * (g + wd * x)
x -= 0.1 * v
print(f"337 LARS   step: x={x.round(4).tolist()}, f={f(x):.4f}, local_lr={local_lr:.3f}")
