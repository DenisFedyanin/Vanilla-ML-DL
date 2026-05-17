"""Раздел 33 — Инициализация (расширенно). Файл 1: базовые.

Концепция 376: Zeros init.
W = 0. Для bias — норм. Для матриц весов — катастрофа: все нейроны
одинаковые, симметрия не нарушается, сеть не учится.

Концепция 377: Ones init.
W = 1. То же — все нейроны одинаковые. Используют только для bias
в некоторых случаях (LSTM forget gate).

Концепция 378: Constant init.
W = c. Любая константа. Как и Zeros/Ones — ломает симметрию только для bias.

Концепция 379: Uniform(a, b).
W ~ U(a, b). Простой случайный диапазон. Базовая нерациональная инициализация.

Концепция 380: Normal(mu, sigma).
W ~ N(mu, sigma). Тоже простая. При неправильной sigma — взрыв или затухание.

Концепция 381: TruncatedNormal (rejection).
Нормальное распределение, обрезанное в [-2sigma, 2sigma]. Не даёт выбросов.
Реализуем перевыборкой (rejection sampling).
"""
import numpy as np

rng = np.random.default_rng(0)
shape = (5, 4)

# === 376: Zeros ===
W = np.zeros(shape)
print(f"376 Zeros: mean={W.mean():.3f}, std={W.std():.3f} — ломает обучение для весов")

# === 377: Ones ===
W = np.ones(shape)
print(f"377 Ones:  mean={W.mean():.3f}, std={W.std():.3f}")

# === 378: Constant ===
W = np.full(shape, 0.5)
print(f"378 Constant(0.5): mean={W.mean():.3f}")

# === 379: Uniform(-0.1, 0.1) ===
W = rng.uniform(-0.1, 0.1, size=shape)
print(f"379 Uniform(-0.1,0.1): mean={W.mean():.4f}, std={W.std():.4f}")

# === 380: Normal(0, 0.1) ===
W = rng.normal(0.0, 0.1, size=shape)
print(f"380 Normal(0,0.1): mean={W.mean():.4f}, std={W.std():.4f}")

# === 381: TruncatedNormal (rejection sampling) ===
def truncated_normal(shape, mu=0.0, sigma=1.0, lo=-2.0, hi=2.0, rng=rng):
    n = int(np.prod(shape))
    out = np.empty(n)
    filled = 0
    while filled < n:
        cand = rng.normal(mu, sigma, size=n)
        ok = cand[(cand >= lo * sigma + mu) & (cand <= hi * sigma + mu)]
        take = min(len(ok), n - filled)
        out[filled:filled + take] = ok[:take]
        filled += take
    return out.reshape(shape)

W = truncated_normal(shape, sigma=1.0)
print(f"381 TruncNormal(sigma=1, |w|<=2): max|w|={np.abs(W).max():.3f}, std={W.std():.3f}")
print(f"    Сравнить с обычным Normal(sigma=1): max|w|≈{np.abs(rng.normal(size=1000)).max():.3f}")
