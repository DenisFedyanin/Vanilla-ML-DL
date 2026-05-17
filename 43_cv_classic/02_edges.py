"""Раздел 43 — Классическое CV.

Файл 2: детекторы границ.

Концепция: Sobel X/Y.
Ядра 3x3, аппроксимирующие частные производные. Sx — вертикальные границы,
Sy — горизонтальные. |grad| = sqrt(Sx^2 + Sy^2), направление = atan2(Sy, Sx).

Концепция: Prewitt.
Похож на Sobel, но без удвоения центральной строки/столбца.
Чуть менее устойчив к шуму, чем Sobel.

Концепция: Scharr.
Улучшенный Sobel с весами 3, 10, 3 — лучше изотропность гради ента.

Концепция: Laplacian.
Вторая производная: подсвечивает зоны быстрого изменения яркости.
Ядро [[0,1,0],[1,-4,1],[0,1,0]] (4-связное) или 8-связное.

Концепция: Canny pipeline.
1) Сгладить Gaussian. 2) Градиент Sobel. 3) Non-maximum suppression вдоль
направления градиента. 4) Двойной порог. 5) Hysteresis: слабые края, связанные
с сильными, остаются.
"""
import numpy as np


def conv2d_same(img, k):
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    p = np.pad(img, ((ph, ph), (pw, pw)), mode="edge")
    out = np.zeros_like(img, dtype=float)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = (p[i:i + kh, j:j + kw] * k).sum()
    return out


rng = np.random.default_rng(0)
# Простое изображение со ступенькой
img = np.zeros((12, 12))
img[:, 6:] = 1.0
img += rng.normal(0, 0.05, img.shape)

sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = sobel_x.T
gx = conv2d_same(img, sobel_x)
gy = conv2d_same(img, sobel_y)
grad = np.sqrt(gx ** 2 + gy ** 2)
print("Sobel: max|gx| на границе =", gx.max().round(2),
      "| ср |grad| =", grad.mean().round(2))

prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
out_prewitt = conv2d_same(img, prewitt_x)
print("Prewitt max:", out_prewitt.max().round(2))

scharr_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
out_scharr = conv2d_same(img, scharr_x)
print("Scharr max:", out_scharr.max().round(2))

laplace = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
out_lap = conv2d_same(img, laplace)
print("Laplacian min/max:", out_lap.min().round(2), out_lap.max().round(2))

# Canny pipeline на той же ступеньке
# 1) Gaussian smooth
g1d = np.exp(-np.arange(-1, 2) ** 2 / 2)
g1d /= g1d.sum()
G = np.outer(g1d, g1d)
sm = conv2d_same(img, G)
# 2) градиент
gx = conv2d_same(sm, sobel_x)
gy = conv2d_same(sm, sobel_y)
mag = np.sqrt(gx ** 2 + gy ** 2)
ang = np.arctan2(gy, gx)
# 3) NMS: упрощённо — оставляем пиксели > обоих горизонтальных соседей (для вертикальных границ)
nms = np.zeros_like(mag)
H, W = mag.shape
for i in range(1, H - 1):
    for j in range(1, W - 1):
        a = ang[i, j]
        # выбираем двух соседей по направлению градиента (квантовано до 4 ориентаций)
        if -np.pi / 8 <= a < np.pi / 8 or a >= 7 * np.pi / 8 or a < -7 * np.pi / 8:
            n1, n2 = mag[i, j - 1], mag[i, j + 1]
        elif np.pi / 8 <= a < 3 * np.pi / 8:
            n1, n2 = mag[i - 1, j + 1], mag[i + 1, j - 1]
        elif 3 * np.pi / 8 <= a < 5 * np.pi / 8:
            n1, n2 = mag[i - 1, j], mag[i + 1, j]
        else:
            n1, n2 = mag[i - 1, j - 1], mag[i + 1, j + 1]
        if mag[i, j] >= n1 and mag[i, j] >= n2:
            nms[i, j] = mag[i, j]
# 4) двойной порог
hi, lo = 0.5 * nms.max(), 0.2 * nms.max()
strong = nms >= hi
weak = (nms >= lo) & (nms < hi)
# 5) hysteresis: слабые рядом с сильными становятся сильными
edges = strong.copy()
for _ in range(3):
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            if weak[i, j] and edges[i - 1:i + 2, j - 1:j + 2].any():
                edges[i, j] = True
print("Canny: число пикселей-границ:", edges.sum(), "из", H * W)
