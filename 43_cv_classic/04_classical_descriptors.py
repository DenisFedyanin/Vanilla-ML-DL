"""Раздел 43 — Классическое CV.

Файл 4: классические дескрипторы.

Концепция: HOG (Histogram of Oriented Gradients).
Делим окно на ячейки. В каждой строим гистограмму ориентаций градиента
(9 бинов, 0..180). Блоки 2x2 ячеек нормируем (L2). Робастен к освещению.
Использовался в детекторе пешеходов Dalal-Triggs.

Концепция: LBP (Local Binary Pattern).
Каждый пиксель сравнивается с 8 соседями: 1 если сосед >= центр, иначе 0.
Получаем 8-битное число 0..255 — текстурный код. Гистограмма LBP — дескриптор.

Концепция: Haar-like features.
Простые прямоугольные фильтры: разности сумм светлых и тёмных областей.
Считаются за O(1) через integral image. Базовая часть Viola-Jones.

Концепция: Integral image.
ii(x,y) = сумма всех пикселей до (x,y). Сумма прямоугольника считается
за 4 обращения: A - B - C + D.

Концепция: Viola-Jones cascade.
Каскад слабых классификаторов на Haar-фичах. Каждый этап быстро отбрасывает
не-объект; финал — точная классификация. Был стандартом face detection до DL.
"""
import numpy as np

rng = np.random.default_rng(0)
img = rng.uniform(0, 1, size=(16, 16))

# --- HOG ---
def hog(img, cell=4, n_bins=9):
    H, W = img.shape
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ang = (np.arctan2(gy, gx) * 180 / np.pi) % 180  # [0,180)
    bin_w = 180 / n_bins
    nh, nw = H // cell, W // cell
    cells = np.zeros((nh, nw, n_bins))
    for i in range(nh):
        for j in range(nw):
            m = mag[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell]
            a = ang[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell]
            bidx = (a // bin_w).astype(int) % n_bins
            for k in range(n_bins):
                cells[i, j, k] = m[bidx == k].sum()
    # блок-нормализация 2x2
    feats = []
    for i in range(nh - 1):
        for j in range(nw - 1):
            block = cells[i:i + 2, j:j + 2].ravel()
            block = block / (np.linalg.norm(block) + 1e-6)
            feats.append(block)
    return np.concatenate(feats)


h_feat = hog(img)
print("HOG: длина дескриптора =", len(h_feat), "| L2 =", np.linalg.norm(h_feat).round(3))

# --- LBP ---
def lbp(img):
    H, W = img.shape
    code = np.zeros((H - 2, W - 2), dtype=np.uint8)
    # 8 соседей в фиксированном порядке (по часовой)
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
            (1, 1), (1, 0), (1, -1), (0, -1)]
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            c = img[i, j]
            v = 0
            for b, (dy, dx) in enumerate(offs):
                if img[i + dy, j + dx] >= c:
                    v |= (1 << b)
            code[i - 1, j - 1] = v
    return code


lbp_code = lbp(img)
lbp_hist, _ = np.histogram(lbp_code, bins=16, range=(0, 256))
print("LBP: гистограмма (16 бинов):", lbp_hist)

# --- Integral image + Haar-like ---
def integral(img):
    return img.cumsum(axis=0).cumsum(axis=1)


def rect_sum(ii, y1, x1, y2, x2):
    """Сумма по [y1..y2, x1..x2] включительно через integral image."""
    A = ii[y2, x2]
    B = ii[y1 - 1, x2] if y1 > 0 else 0
    C = ii[y2, x1 - 1] if x1 > 0 else 0
    D = ii[y1 - 1, x1 - 1] if y1 > 0 and x1 > 0 else 0
    return A - B - C + D


ii = integral(img)
# Haar-like 2-rect (горизонтальный): чёрный минус белый
haar = rect_sum(ii, 2, 2, 5, 7) - rect_sum(ii, 2, 8, 5, 13)
print("Haar 2-rect feature:", haar.round(3))
# Проверка
direct = img[2:6, 2:8].sum() - img[2:6, 8:14].sum()
print("Прямая сумма:", direct.round(3), "| совпадает:", np.isclose(haar, direct))

# Viola-Jones cascade — концепт
print("Viola-Jones: каскад из ~38 этапов; ранний reject ускоряет в ~15 раз")
