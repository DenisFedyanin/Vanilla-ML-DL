"""Раздел 28 — Слои нейронных сетей. Файл 4: пулинг.

Концепция 293: MaxPool 2x2.
В каждом окне 2x2 берём максимум. Снижает разрешение в 2 раза.
Делает признаки инвариантными к малым сдвигам. Классика CNN.

Концепция 294: AvgPool 2x2.
В каждом окне 2x2 берём среднее. Сглаживает, не выделяет резкие пики.

Концепция 295: Global Average Pooling (GAP).
Среднее по всем пространственным позициям -> вектор размера C.
Заменяет огромный Flatten+Dense в классификаторах (ResNet head).

Концепция 296: Adaptive pooling.
Пулим в фиксированный выходной размер (H_out, W_out) независимо от входа.
Делается через переменный размер окна. Часто на стыке с Dense-головой.
"""
import numpy as np

rng = np.random.default_rng(0)

# === 293: MaxPool 2x2 ===
def maxpool2x2(x):
    H, W = x.shape
    out = np.zeros((H // 2, W // 2))
    for i in range(H // 2):
        for j in range(W // 2):
            out[i, j] = x[2 * i:2 * i + 2, 2 * j:2 * j + 2].max()
    return out

img = rng.standard_normal((4, 4))
y_max = maxpool2x2(img)
print(f"293 MaxPool: {img.shape} -> {y_max.shape}")
print(f"    Пример окна {img[:2, :2].round(2).tolist()} -> max={y_max[0, 0]:.2f}")

# === 294: AvgPool 2x2 ===
def avgpool2x2(x):
    H, W = x.shape
    out = np.zeros((H // 2, W // 2))
    for i in range(H // 2):
        for j in range(W // 2):
            out[i, j] = x[2 * i:2 * i + 2, 2 * j:2 * j + 2].mean()
    return out

y_avg = avgpool2x2(img)
print(f"294 AvgPool: {img.shape} -> {y_avg.shape}, окно среднее={y_avg[0, 0]:.2f}")

# === 295: Global Average Pooling ===
def gap(x):
    # x: (C, H, W) -> (C,)
    return x.mean(axis=(1, 2))

feat = rng.standard_normal((8, 6, 6))
g = gap(feat)
print(f"295 GAP: {feat.shape} -> {g.shape}")

# === 296: Adaptive pooling (fix output size) ===
def adaptive_avg_pool2d(x, out_h, out_w):
    H, W = x.shape
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            i0, i1 = (i * H) // out_h, ((i + 1) * H) // out_h
            j0, j1 = (j * W) // out_w, ((j + 1) * W) // out_w
            out[i, j] = x[i0:i1, j0:j1].mean()
    return out

big = rng.standard_normal((7, 7))
ad = adaptive_avg_pool2d(big, 3, 3)
print(f"296 Adaptive pool: 7x7 -> 3x3 (окна не одинаковые!)")
print(f"    out={ad.round(2).tolist()}")
