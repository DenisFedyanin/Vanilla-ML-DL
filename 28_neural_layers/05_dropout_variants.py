"""Раздел 28 — Слои нейронных сетей. Файл 5: dropout-варианты.

Концепция 297: Vanilla Dropout.
С вероятностью p обнуляем нейрон (умножаем на бернулли-маску).
Активные нейроны масштабируются на 1/(1-p) — inverted dropout, чтобы
матожидание не сдвигалось. На eval-режиме выключен.

Концепция 298: SpatialDropout (Drop channels).
В CNN обнуляются целые карты признаков (каналы) целиком.
Поскольку соседние пиксели сильно коррелируют, обычный dropout
почти не регуляризует. SpatialDropout эффективнее.

Концепция 299: AlphaDropout.
Спец dropout для SELU-сетей: сохраняет среднее=0 и дисперсию=1.
Зануляет с заменой на спец. значение и аффинно нормирует — поддерживает
свойство self-normalization.

Концепция 300: Scheduled dropout.
Скорость p меняется по ходу обучения: например, линейно растёт
от 0 до p_max или падает. Стандартный приём для стабилизации.
"""
import numpy as np

rng = np.random.default_rng(0)

# === 297: Vanilla dropout (inverted) ===
def dropout(x, p, rng, train=True):
    if not train or p == 0:
        return x
    mask = (rng.uniform(size=x.shape) > p).astype(float)
    return x * mask / (1 - p)

x = rng.standard_normal(10_000)
y = dropout(x, p=0.3, rng=rng)
print(f"297 Dropout p=0.3: исх mean={x.mean():.3f}, после={y.mean():.3f} (д.б. близко)")
print(f"    исх var={x.var():.3f}, после var={y.var():.3f} (выше из-за маски)")

# === 298: SpatialDropout (drop entire channels) ===
def spatial_dropout(x, p, rng, train=True):
    # x: (B, C, H, W)
    if not train or p == 0:
        return x
    B, C, H, W = x.shape
    mask = (rng.uniform(size=(B, C, 1, 1)) > p).astype(float)
    return x * mask / (1 - p)

feat = rng.standard_normal((2, 4, 3, 3))
y_sp = spatial_dropout(feat, p=0.5, rng=rng)
# каналы либо все нули, либо все умножены на 2
zero_channels = (y_sp.reshape(2, 4, -1).sum(axis=2) == 0).sum()
print(f"298 SpatialDropout: занулено каналов={zero_channels}/{2 * 4}")

# === 299: AlphaDropout (концепт для SELU) ===
def alpha_dropout(x, p, rng, train=True):
    if not train or p == 0:
        return x
    # фиксированное "альфа" из теории SELU
    alpha = -1.7580993408473766
    mask = (rng.uniform(size=x.shape) > p).astype(float)
    a = ((1 - p) * (1 + p * alpha ** 2)) ** -0.5
    b = -a * alpha * p
    return a * (x * mask + alpha * (1 - mask)) + b

x = rng.standard_normal(10_000)
y_a = alpha_dropout(x, p=0.1, rng=rng)
print(f"299 AlphaDropout: mean={y_a.mean():.3f} (~0), std={y_a.std():.3f} (~1) — сохраняется самонормализация")

# === 300: Scheduled dropout ===
def linear_schedule(epoch, total, p_max):
    return p_max * (epoch / total)

epochs = 5
schedule = [linear_schedule(e, epochs, p_max=0.5) for e in range(epochs + 1)]
print(f"300 Scheduled dropout p по эпохам: {[round(s, 2) for s in schedule]}")
