"""Раздел 62 — Трюки обучения.

Файл 2: Манипуляции с градиентами.

Концепция: Gradient clipping by norm.
Если ||g|| > C, то g <- g * C / ||g||. Защищает от взрывающихся градиентов в RNN.

Концепция: Gradient clipping by value.
g <- clip(g, -C, C). Поэлементно. Проще, но искажает направление.

Концепция: Gradient noise.
Добавляем G_i ~ N(0, sigma^2) к grad. sigma часто уменьшается по расписанию sigma_t = eta/(1+t)^gamma.
Помогает выбираться из плохих локальных минимумов.

Концепция: Lookahead optimizer.
Поддерживает "медленные" веса phi и "быстрые" theta. Каждые k шагов inner-оптимизатора:
phi <- phi + alpha (theta - phi), theta <- phi. Стабильнее и часто сходится быстрее.
"""
import numpy as np

rng = np.random.default_rng(0)

# === Clipping by norm ===
g = rng.normal(size=5) * 10
C = 1.0
norm = np.linalg.norm(g)
g_clip = g * (C / norm) if norm > C else g
print(f"||g||={norm:.3f}, после clip_norm({C}): ||g_clip||={np.linalg.norm(g_clip):.3f}")

# === Clipping by value ===
g_val = np.clip(g, -C, C)
print(f"clip_value({C}): {g_val.round(3)}")

# === Gradient noise: добавляем шум, уменьшающийся со временем ===
g_clean = np.array([1.0, -0.5, 0.3])
for t in [0, 10, 100]:
    sigma = 0.1 / (1 + t) ** 0.55
    g_noisy = g_clean + sigma * rng.normal(size=3)
    print(f"t={t}: sigma={sigma:.4f}, g_noisy={g_noisy.round(3)}")

# === Lookahead на простой задаче: min ||x - x*||^2 ===
x_star = np.array([2.0, -1.0])
def grad(x): return 2 * (x - x_star)

# Базовый inner SGD
theta = np.zeros(2); lr = 0.3
for _ in range(20):
    theta = theta - lr * grad(theta)
print(f"Inner SGD: theta={theta.round(4)}")

# Lookahead: phi обновляется каждые k=5 шагов
theta = np.zeros(2); phi = theta.copy()
k, alpha = 5, 0.5
for t in range(1, 21):
    theta = theta - lr * grad(theta)
    if t % k == 0:
        phi = phi + alpha * (theta - phi)
        theta = phi.copy()
print(f"Lookahead: phi={phi.round(4)}")
