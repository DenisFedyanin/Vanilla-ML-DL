"""Раздел 31 — Лоссы (расширенно). Файл 4: metric learning.

Концепция 355: Triplet loss.
max(0, d(a, p) - d(a, n) + margin). Хотим: расстояние до позитива
меньше, чем до негатива, минимум на margin.

Концепция 356: Contrastive loss (Hadsell et al.).
y*d^2 + (1-y)*max(0, m - d)^2. Если пара похожая (y=1) — стягиваем,
если разная — отталкиваем не ближе margin.

Концепция 357: InfoNCE / NT-Xent.
-log( exp(sim(a,p)/tau) / sum_{neg} exp(sim(a,neg)/tau) ).
Контрастная задача: позитив vs N негативов как softmax-classification.
Сердце SimCLR, MoCo.

Концепция 358: Cosine embedding loss.
y=1: 1 - cos(x1,x2); y=-1: max(0, cos(x1,x2)-margin).
Использует косинус — нечувствительно к длине векторов.
"""
import numpy as np

rng = np.random.default_rng(0)
d = 4
def norm(v, axis=-1, keepdims=True): return np.linalg.norm(v, axis=axis, keepdims=keepdims)

# Сделаем 3 группы похожих эмбеддингов
a = rng.standard_normal((4, d))
p = a + 0.1 * rng.standard_normal((4, d))    # позитив рядом
n = rng.standard_normal((4, d)) * 2          # негатив подальше

# === 355: Triplet loss ===
def triplet(a, p, n, margin=1.0):
    dp = np.linalg.norm(a - p, axis=1)
    dn = np.linalg.norm(a - n, axis=1)
    return np.maximum(0, dp - dn + margin).mean()

print(f"355 Triplet margin=1.0 = {triplet(a, p, n):.4f}")

# === 356: Contrastive loss ===
def contrastive(x1, x2, y, margin=1.0):
    d = np.linalg.norm(x1 - x2, axis=1)
    return (y * d ** 2 + (1 - y) * np.maximum(0, margin - d) ** 2).mean()

# 2 похожих пары + 2 разных
x1 = np.vstack([a[:2], a[2:]])
x2 = np.vstack([p[:2], n[2:]])
y_pair = np.array([1, 1, 0, 0])
print(f"356 Contrastive = {contrastive(x1, x2, y_pair):.4f}")

# === 357: InfoNCE / NT-Xent ===
def info_nce(anchors, positives, temperature=0.1):
    A = anchors / norm(anchors); P = positives / norm(positives)
    sims = A @ P.T / temperature       # (N, N)
    labels = np.arange(len(A))         # позитив на диагонали
    s = sims - sims.max(axis=1, keepdims=True)
    log_sm = s - np.log(np.exp(s).sum(axis=1, keepdims=True))
    return -log_sm[np.arange(len(A)), labels].mean()

print(f"357 InfoNCE = {info_nce(a, p):.4f}  (низкое = anchor ближе к своему позитиву)")
print(f"    InfoNCE на случайных парах a-n = {info_nce(a, n):.4f} (выше)")

# === 358: Cosine embedding loss ===
def cosine_emb_loss(x1, x2, y, margin=0.0):
    c = (x1 * x2).sum(axis=1) / (norm(x1, keepdims=False) * norm(x2, keepdims=False))
    pos = 1 - c
    neg = np.maximum(0, c - margin)
    return np.where(y == 1, pos, neg).mean()

print(f"358 Cosine emb loss = {cosine_emb_loss(x1, x2, y_pair):.4f}")
