"""Раздел 65 — Metric learning.

Файл 1: Siamese & Triplet loss.

Концепция: Siamese networks.
Две одинаковые сети с общими весами получают пару объектов (x1, x2) и выдают
эмбеддинги f(x1), f(x2). Цель — обучить расстояние ||f(x1)-f(x2)||, чтобы
одинаковые классы были близко, разные далеко. Применение: верификация лиц/подписей.

Концепция: Contrastive loss (Hadsell, 2006).
L = y * d^2 + (1-y) * max(0, m - d)^2, где d = ||f(x1)-f(x2)||, y=1 если пара
'позитивная' (тот же класс). Тянем позитивные вместе, отталкиваем негативные
за пределы маржи m.

Концепция: Triplet loss (FaceNet).
Тройка (anchor, positive, negative). Лосс:
L = max(0, ||f(a) - f(p)||^2 - ||f(a) - f(n)||^2 + m).
Хотим, чтобы расстояние до позитива было меньше, чем до негатива хотя бы на m.

Концепция: Easy / Hard / Semi-hard negatives.
- easy negative: ||a-n|| уже > ||a-p|| + m, лосс=0, бесполезен.
- hard negative: ||a-n|| < ||a-p||, очень информативно, но может развалить обучение.
- semi-hard: ||a-p|| < ||a-n|| < ||a-p|| + m. Золотая середина — FaceNet берёт именно их.

Концепция: Semi-hard mining (стратегия).
В минибатче для каждого (a,p) ищем все n с ||a-n|| в полосе (||a-p||, ||a-p||+m)
и берём ближайший такой. Если нет — берём самого ближнего из 'не-проще-чем-p'.

Демо: вручную считаем triplet loss и иллюстрируем три типа негативов.
"""
import numpy as np

rng = np.random.default_rng(0)

# === игрушечные эмбеддинги: 3 класса в 2D ===
def emb(c, n):
    centers = np.array([[0, 0], [3, 0], [0, 3]], dtype=float)
    return rng.normal(centers[c], 0.5, size=(n, 2))

a = emb(0, 1)[0]          # anchor класса 0
p = emb(0, 1)[0]          # positive того же класса
negs = emb(1, 5)          # 5 negatives класса 1

# === Triplet loss с маржой ===
margin = 1.0
d_ap = np.linalg.norm(a - p)
d_an = np.linalg.norm(a - negs, axis=1)
losses = np.maximum(0, d_ap ** 2 - d_an ** 2 + margin)
print(f"d(a,p)={d_ap:.2f}, d(a,n)={np.round(d_an, 2)}")
print(f"Triplet losses per negative: {np.round(losses, 3)}")
print(f"Среднее лоссов: {losses.mean():.3f}")

# === Классификация негативов ===
easy = d_an > d_ap + margin
hard = d_an < d_ap
semi_hard = (d_an >= d_ap) & (d_an < d_ap + margin)
print(f"easy={easy.sum()}, hard={hard.sum()}, semi-hard={semi_hard.sum()}")

# === Semi-hard mining ===
if semi_hard.any():
    idx = np.where(semi_hard)[0]
    best = idx[np.argmin(d_an[idx])]
    print(f"Semi-hard mining: выбрали negative #{best}, d_an={d_an[best]:.2f}")
else:
    print("Semi-hard mining: подходящих не нашлось, fallback на hard")

# === Contrastive loss на парах ===
def contrastive(f1, f2, y, m=1.0):
    d = np.linalg.norm(f1 - f2, axis=-1)
    return y * d ** 2 + (1 - y) * np.maximum(0, m - d) ** 2

pair_pos = (emb(0, 1)[0], emb(0, 1)[0], 1)
pair_neg = (emb(0, 1)[0], emb(1, 1)[0], 0)
L_pos = contrastive(pair_pos[0], pair_pos[1], pair_pos[2])
L_neg = contrastive(pair_neg[0], pair_neg[1], pair_neg[2])
print(f"Contrastive loss: pos={L_pos:.3f}, neg={L_neg:.3f}")

# === Простой шаг градиентного спуска по triplet ===
# d/da [||a-p||^2 - ||a-n||^2] = 2(p_a - p) - 2(a - n) = 2(n - p)
a_new = a.copy()
n0 = negs[0]
for _ in range(50):
    loss = max(0, np.sum((a_new - p) ** 2) - np.sum((a_new - n0) ** 2) + margin)
    if loss == 0:
        break
    grad = 2 * (n0 - p)
    a_new -= 0.05 * grad
print(f"После 50 шагов: d(a,p)={np.linalg.norm(a_new - p):.2f}, d(a,n)={np.linalg.norm(a_new - n0):.2f}")
