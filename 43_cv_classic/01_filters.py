"""Раздел 43 — Классическое CV.

Файл 1: фильтры размытия.

Концепция: Box blur.
Среднее по окну KxK. Простой, но даёт ореолы и не сглаживает диагонали хорошо.

Концепция: Gaussian blur.
Веса по нормальному распределению — гладко, без ореолов. Сепарабельный:
делается двумя 1D-свёртками вдоль X и Y, что в K раз быстрее.

Концепция: Median filter.
В каждом окне берём медиану. Отлично убирает соль-перец шум, сохраняет
края (в отличие от линейных фильтров). Не сепарабельный — нелинейный.

Концепция: Bilateral filter.
Веса зависят и от расстояния (как у Gaussian), и от разницы интенсивности:
w = G_s(dist) * G_r(|I(p)-I(q)|). Сглаживает плоские области, сохраняя края.
"""
import numpy as np


def conv2d_same(img, k):
    """2D свёртка с 'same' паддингом."""
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="edge")
    out = np.zeros_like(img, dtype=float)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = (padded[i:i + kh, j:j + kw] * k).sum()
    return out


rng = np.random.default_rng(0)
img = rng.uniform(0, 1, size=(12, 12))
# Добавим соль-перец шум
noise_mask = rng.uniform(size=img.shape) < 0.1
img_noisy = img.copy()
img_noisy[noise_mask] = rng.choice([0.0, 1.0], size=noise_mask.sum())

# Box blur
box = np.ones((3, 3)) / 9
out_box = conv2d_same(img_noisy, box)
print("Box blur: ср =", out_box.mean().round(3), "std =", out_box.std().round(3))

# Gaussian blur (сепарабельный)
sigma = 1.0
xs = np.arange(-2, 3)
g1d = np.exp(-xs ** 2 / (2 * sigma ** 2))
g1d /= g1d.sum()
G = np.outer(g1d, g1d)
out_g = conv2d_same(img_noisy, G)
print("Gaussian blur std:", out_g.std().round(3), "(меньше, чем у box)")

# Median filter
def median_filter(img, k=3):
    p = k // 2
    padded = np.pad(img, p, mode="edge")
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = np.median(padded[i:i + k, j:j + k])
    return out


out_med = median_filter(img_noisy, 3)
# Сравним удаление шума: ошибка относительно чистого
err_box = np.abs(out_box - img).mean()
err_med = np.abs(out_med - img).mean()
print(f"MAE после box={err_box:.3f}, после median={err_med:.3f} (median лучше для соли-перца)")

# Bilateral filter (тиновая реализация)
def bilateral(img, k=3, sigma_s=1.0, sigma_r=0.2):
    p = k // 2
    padded = np.pad(img, p, mode="edge")
    out = np.zeros_like(img)
    yy, xx = np.mgrid[-p:p + 1, -p:p + 1]
    spatial = np.exp(-(yy ** 2 + xx ** 2) / (2 * sigma_s ** 2))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            patch = padded[i:i + k, j:j + k]
            center = padded[i + p, j + p]
            range_w = np.exp(-(patch - center) ** 2 / (2 * sigma_r ** 2))
            w = spatial * range_w
            out[i, j] = (patch * w).sum() / w.sum()
    return out


out_bil = bilateral(img_noisy, k=3, sigma_s=1.0, sigma_r=0.2)
err_bil = np.abs(out_bil - img).mean()
print(f"MAE после bilateral={err_bil:.3f} (сохраняет края)")
