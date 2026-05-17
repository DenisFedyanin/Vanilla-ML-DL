"""Раздел 62 — Трюки обучения.

Файл 4: Усреднение весов.

Концепция: EMA (Exponential Moving Average).
theta_ema <- mu * theta_ema + (1-mu) * theta. mu близок к 1 (например 0.999).
EMA-веса часто дают лучше валидацию (более "плоский" минимум).

Концепция: SWA (Stochastic Weight Averaging).
Простое арифметическое среднее последних K снапшотов модели (обычно с конца обучения).
theta_swa = (1/K) sum theta_k. Похоже на EMA, но равные веса.

Концепция: Polyak averaging.
Среднее всех итераций SGD: theta_avg_t = (1/t) sum_{k<=t} theta_k.
Уменьшает шум стохастического градиента. Доказательно ускоряет SGD на выпуклых задачах.
"""
import numpy as np

rng = np.random.default_rng(0)

# Будем тренироваться на f(w) = 0.5 ||w - w*||^2, шумный grad
w_star = np.array([3.0, -1.5])
n_steps = 200

def step(w, lr=0.05):
    g = (w - w_star) + 0.5 * rng.normal(size=2)
    return w - lr * g

# === EMA ===
w = np.zeros(2); ema = w.copy(); mu = 0.99
for t in range(n_steps):
    w = step(w)
    ema = mu * ema + (1 - mu) * w
print(f"Финальный SGD: w={w.round(4)}, EMA: {ema.round(4)}, true: {w_star}")

# === SWA: усредним последние 50 снапшотов с шагом 5 ===
rng = np.random.default_rng(0)
w = np.zeros(2); snapshots = []
for t in range(n_steps):
    w = step(w)
    if t >= n_steps - 50 and t % 5 == 0:
        snapshots.append(w.copy())
swa = np.mean(snapshots, axis=0)
print(f"SWA (по {len(snapshots)} снапшотам): {swa.round(4)}")

# === Polyak: среднее всех ===
rng = np.random.default_rng(0)
w = np.zeros(2); s = np.zeros(2)
for t in range(n_steps):
    w = step(w); s += w
polyak = s / n_steps
print(f"Polyak: {polyak.round(4)}")

# Сравним ошибки
print("Ошибки до w*:")
for name, val in [("last w", w), ("EMA", ema), ("SWA", swa), ("Polyak", polyak)]:
    print(f"  {name:8s}: ||.-w*||={np.linalg.norm(val - w_star):.4f}")
