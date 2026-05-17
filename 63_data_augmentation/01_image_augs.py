"""Раздел 63 — Data augmentation.

Файл 1: Базовые image-аугментации.

Концепция: Flip H/V.
Зеркальное отражение по оси. Подходит для естественных изображений (кошки/собаки),
не подходит для текста/цифр.

Концепция: Rotate 90/180/270.
Дискретные повороты без интерполяции. Применяется когда ориентация не информативна
(аэросъемка, микроскопия).

Концепция: Random crop.
Случайно вырезаем подобласть и (опционально) ресайзим обратно. Учит инвариантности
к позиции и масштабу.

Концепция: Color jitter.
Шум по яркости/контрасту/насыщенности. Простейшая версия — additive Gaussian per-channel.

Концепция: Gaussian blur.
Свертка с гауссовым ядром sigma. Симулирует расфокус, заставляет модель меньше
полагаться на высокие частоты.

Концепция: Cutout.
Зануляем (или заменяем средним) прямоугольную область. Регуляризатор, подобие dropout.
"""
import numpy as np

rng = np.random.default_rng(0)

# Создадим тестовое "изображение" 16x16x3
img = (rng.uniform(0, 1, (16, 16, 3)) * 255).astype(np.uint8)

# === Flip ===
flip_h = np.flip(img, axis=1)
flip_v = np.flip(img, axis=0)
print(f"Flip H: shape={flip_h.shape}, изменилось: {not np.array_equal(flip_h, img)}")
print(f"Flip V: pixel(0,0) до/после = {img[0,0,0]} vs {flip_v[0,0,0]}")

# === Rotate (дискретные) ===
r90 = np.rot90(img, k=1)
r180 = np.rot90(img, k=2)
r270 = np.rot90(img, k=3)
print(f"rot90: {r90.shape}, rot180: {r180.shape}, rot270: {r270.shape}")

# === Random crop 12x12 ===
crop_size = 12
y0 = rng.integers(0, img.shape[0] - crop_size + 1)
x0 = rng.integers(0, img.shape[1] - crop_size + 1)
crop = img[y0:y0 + crop_size, x0:x0 + crop_size]
print(f"RandomCrop: origin=({y0},{x0}), shape={crop.shape}")

# === Color jitter: добавим шум разной интенсивности по каналам ===
jitter = img.astype(float) + rng.normal(0, 10, (1, 1, 3))
jitter = np.clip(jitter, 0, 255).astype(np.uint8)
print(f"ColorJitter: mean до={img.mean():.2f}, после={jitter.mean():.2f}")

# === Gaussian blur 3x3 ядром ===
def gauss_kernel(size=3, sigma=1.0):
    ax = np.arange(size) - size // 2
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    k1d = g / g.sum()
    return np.outer(k1d, k1d)

def conv2d(img, k):
    h, w = img.shape[:2]; kh, kw = k.shape
    pad_y, pad_x = kh // 2, kw // 2
    padded = np.pad(img, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode='edge')
    out = np.zeros_like(img, dtype=float)
    for i in range(h):
        for j in range(w):
            patch = padded[i:i + kh, j:j + kw, :]
            out[i, j] = (patch * k[:, :, None]).sum(axis=(0, 1))
    return out.astype(img.dtype)

blurred = conv2d(img, gauss_kernel(3, 1.0))
print(f"Blur: max diff с оригиналом = {np.abs(blurred.astype(int) - img.astype(int)).max()}")

# === Cutout: занулить прямоугольник 5x5 ===
cut = img.copy()
y0, x0 = rng.integers(0, 12, size=2)
cut[y0:y0 + 5, x0:x0 + 5] = 0
print(f"Cutout: занулили 25 пикселей, mean: {img.mean():.2f} -> {cut.mean():.2f}")
