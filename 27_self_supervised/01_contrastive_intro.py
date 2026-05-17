"""Раздел 27 — Self-supervised learning. Файл 1: введение в contrastive.

Концепция 703: Self-supervised learning (SSL) — идея.
Учим представления БЕЗ разметки, придумывая «pretext»-задачу из самих данных:
угадать поворот картинки, заполнить пропуск, сравнить аугментации и т.п.
Затем готовые представления переносим на downstream-задачи.

Концепция 704: Augmentation pairs (positive pair).
Из одного объекта x делаем две случайные аугментации x_a, x_b. Они образуют
positive pair. Все другие объекты в батче — negatives.

Концепция 705: Contrastive learning.
Цель: представления z_a, z_b близки (positive), а с другими — далеки (negatives).
Меру обычно берут косинусную: sim(u, v) = u^T v / (||u|| ||v||).

Концепция 706: InfoNCE / NT-Xent loss.
L = -log( exp(sim(z_a, z_b)/T) / sum_k exp(sim(z_a, z_k)/T) ),
где сумма по всем z_k в батче (включая negative). T — температура.
Эквивалентно softmax-кросс-энтропии с правильным классом = positive.
"""
import numpy as np

rng = np.random.default_rng(0)

# === 703-704: симулируем 8 объектов и две «аугментации» (шум) ===
N, D = 8, 16
X = rng.normal(size=(N, D))
def augment(x, noise=0.1):
    return x + noise * rng.normal(size=x.shape)

Za = augment(X, 0.1)
Zb = augment(X, 0.1)

# Нормализуем (для cosine similarity)
def l2norm(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)

Za, Zb = l2norm(Za), l2norm(Zb)

# === 705: матрица сходств ===
sim = Za @ Zb.T  # (N, N)
print(f"705 Среднее sim positive (диагональ)={np.mean(np.diag(sim)):.3f}")
neg_mean = (sim.sum() - np.trace(sim)) / (N * N - N)
print(f"     Среднее sim negative (вне диагонали)={neg_mean:.3f}")

# === 706: ручной InfoNCE / NT-Xent ===
T = 0.1
logits = sim / T  # (N, N)
# для каждой строки правильный класс — её же индекс (positive — диагональ)
logits = logits - logits.max(axis=1, keepdims=True)  # численная стабильность
exp = np.exp(logits)
prob = exp / exp.sum(axis=1, keepdims=True)
loss = -np.log(prob[np.arange(N), np.arange(N)] + 1e-12).mean()
print(f"706 InfoNCE loss (T={T}) на хорошо разделимых парах: {loss:.3f}")

# Для сравнения: случайные представления должны давать loss ≈ log(N)
Zr = l2norm(rng.normal(size=(N, D)))
sim_r = Zr @ Zr.T / T
sim_r -= sim_r.max(axis=1, keepdims=True)
expr = np.exp(sim_r)
loss_r = -np.log((expr[np.arange(N), np.arange(N)] / expr.sum(axis=1)) + 1e-12).mean()
print(f"706 На случайных векторах: loss={loss_r:.3f} (≈ log(N)={np.log(N):.2f})")
