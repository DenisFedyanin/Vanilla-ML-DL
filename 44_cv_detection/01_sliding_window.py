"""Раздел 44 — Object Detection.

Файл 1: скользящее окно и template matching.

Концепция: Sliding window.
Перебираем все позиции окна заданного размера, считаем "оценку" (классификатор
или похожесть на шаблон). Базовая идея до RPN-эпохи. Дорого: O(H*W) позиций.

Концепция: Template matching.
Сравниваем шаблон T с фрагментом I. SSD = sum((T-I)^2), CC = sum(T*I).
NCC (Normalized Cross-Correlation) = sum((T-T_mean)(I-I_mean)) /
(sigma_T * sigma_I). Робастен к освещению.

Концепция: Image pyramid.
Многомасштабное представление: исходное изображение, /2, /4, /8... Позволяет
искать объект разного размера одной моделью. Gaussian pyramid: smooth + subsample.
"""
import numpy as np


rng = np.random.default_rng(0)
H, W = 20, 20
img = rng.normal(0, 0.1, size=(H, W))
# Поместим шаблон-кружок в (10, 12)
template = np.zeros((5, 5))
yy, xx = np.mgrid[-2:3, -2:3]
template[(yy ** 2 + xx ** 2) <= 4] = 1.0
img[8:13, 10:15] += template
print("Шаблон размер:", template.shape, "истинная позиция (top-left): (8, 10)")

# Sliding window + NCC
def ncc(patch, T):
    p = patch - patch.mean()
    t = T - T.mean()
    denom = (np.sqrt((p ** 2).sum() * (t ** 2).sum()) + 1e-9)
    return (p * t).sum() / denom


th, tw = template.shape
score = np.full((H - th + 1, W - tw + 1), -np.inf)
for i in range(score.shape[0]):
    for j in range(score.shape[1]):
        score[i, j] = ncc(img[i:i + th, j:j + tw], template)

best = np.unravel_index(np.argmax(score), score.shape)
print("Лучший матч (top-left):", best, "| score =", score[best].round(3))

# SSD для сравнения
ssd = np.zeros_like(score)
for i in range(ssd.shape[0]):
    for j in range(ssd.shape[1]):
        ssd[i, j] = ((img[i:i + th, j:j + tw] - template) ** 2).sum()
best_ssd = np.unravel_index(np.argmin(ssd), ssd.shape)
print("Best SSD позиция:", best_ssd, "| ssd =", ssd[best_ssd].round(3))

# Image pyramid (gaussian + downsample 2x)
def downsample(img):
    # простое 2x усреднение
    h, w = img.shape
    h2, w2 = h // 2, w // 2
    img = img[:h2 * 2, :w2 * 2]
    return img.reshape(h2, 2, w2, 2).mean(axis=(1, 3))


pyramid = [img]
for _ in range(3):
    if min(pyramid[-1].shape) < 4:
        break
    pyramid.append(downsample(pyramid[-1]))
print("Пирамида размеров:", [p.shape for p in pyramid])

# Поиск на каждом уровне (упрощённо: только для уровня 0 и 1)
print("Sliding window O(H*W*K*K) — поэтому в современных детекторах его заменили RPN/anchor-free")
