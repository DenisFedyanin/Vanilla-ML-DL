"""Раздел 43 — Классическое CV.

Файл 3: углы и ключевые точки.

Концепция: Harris corner detector.
Для каждой точки считаем матрицу автокорреляции M = sum_w (grad * grad^T).
Отклик R = det(M) - k * trace(M)^2. R>0 — угол, R<0 — край, R~0 — гладко.

Концепция: FAST corner.
Сравниваем центр с 16 пикселями круга радиуса 3. Если N подряд сильно
ярче/темнее центра — это угол. Очень быстрый, используется в ORB.

Концепция: ORB (Oriented FAST + Rotated BRIEF).
FAST для детекции, BRIEF для описания (бинарный дескриптор: пары пикселей,
больше/меньше). Дешёвая альтернатива SIFT, не запатентована.

Концепция: SIFT.
Scale-Invariant Feature Transform. Detect: DoG (Difference of Gaussians)
на пирамиде масштабов. Descriptor: гистограммы ориентаций градиента в 16
ячейках вокруг точки -> 128-мерный вектор.

Концепция: SURF.
Speed-Up Robust Features. Аппроксимация SIFT через интегральные изображения
и box-фильтры — быстрее, но запатентован.
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
# Изображение с явным углом (квадрат)
img = np.zeros((16, 16))
img[4:12, 4:12] = 1.0

sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sy = sx.T
Ix = conv2d_same(img, sx)
Iy = conv2d_same(img, sy)
Ixx, Iyy, Ixy = Ix * Ix, Iy * Iy, Ix * Iy
# Сумма по окну 3x3
w = np.ones((3, 3))
Sxx = conv2d_same(Ixx, w)
Syy = conv2d_same(Iyy, w)
Sxy = conv2d_same(Ixy, w)
det = Sxx * Syy - Sxy ** 2
trace = Sxx + Syy
k_h = 0.04
R = det - k_h * trace ** 2
# Углы — локальные максимумы R
ys, xs = np.where(R > 0.5 * R.max())
print("Harris: найдено кандидатов в углы:", len(ys),
      "| диапазон позиций X:", xs.min(), "-", xs.max(),
      "| Y:", ys.min(), "-", ys.max())

# FAST corner: упрощённая 8-точечная версия
def fast_corner(img, t=0.2, n_consec=5):
    H, W = img.shape
    offs = [(-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)]
    corners = []
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            c = img[i, j]
            vals = np.array([img[i + dy, j + dx] for dy, dx in offs])
            bright = vals > c + t
            dark = vals < c - t
            # цикличность: проверим, есть ли n_consec True в кольце
            bb = np.concatenate([bright, bright])
            dd = np.concatenate([dark, dark])
            for s in range(8):
                if bb[s:s + n_consec].all() or dd[s:s + n_consec].all():
                    corners.append((i, j))
                    break
    return corners


fast_pts = fast_corner(img, t=0.2, n_consec=5)
print("FAST: число найденных угловых точек:", len(fast_pts))

# ORB descriptor — концепт: BRIEF = бинарные сравнения пар пикселей в окне
def brief_descriptor(img, pt, n_bits=16, win=5):
    rng_b = np.random.default_rng(1)
    i, j = pt
    p = win // 2
    pairs = rng_b.integers(-p, p + 1, size=(n_bits, 4))
    bits = []
    for a in range(n_bits):
        y1, x1, y2, x2 = pairs[a]
        v1 = img[max(0, min(img.shape[0] - 1, i + y1)),
                 max(0, min(img.shape[1] - 1, j + x1))]
        v2 = img[max(0, min(img.shape[0] - 1, i + y2)),
                 max(0, min(img.shape[1] - 1, j + x2))]
        bits.append(1 if v1 < v2 else 0)
    return np.array(bits, dtype=np.uint8)


desc = brief_descriptor(img, fast_pts[0] if fast_pts else (5, 5))
print("BRIEF дескриптор (16 бит) первой точки:", desc)

# SIFT/SURF — только концепт: DoG пирамида и 128-мерный вектор ориентаций
print("SIFT концепт: DoG = G(sigma1) - G(sigma2); 128 = 16 cells * 8 orientations")
print("SURF концепт: integral image + Haar-box -> приближение Hessian")
