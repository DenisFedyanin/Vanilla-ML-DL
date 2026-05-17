"""Раздел 65 — Metric learning.

Файл 2: Лоссы для распознавания лиц.

Концепция: Center loss (Wen et al., 2016).
Дополнительный лосс к softmax: L_c = 1/2 * sum ||x_i - c_{y_i}||^2.
Тянет признаки к центру своего класса, делая кластеры компактными.
Центры обновляются скользящим средним: c_y <- c_y - alpha * (c_y - x_i).

Концепция: Large-Margin Softmax (L-Softmax).
Классический softmax: W_y^T x = ||W_y|| ||x|| cos(theta).
L-Softmax заменяет cos(theta) на cos(m*theta), добавляя угловой margin.
Делает классы более разделимыми по углу.

Концепция: SphereFace (A-Softmax).
То же L-Softmax + нормализация ||W||=1, чтобы решение зависело только от угла.
logit = ||x|| * cos(m*theta_y).

Концепция: CosFace (Large Margin Cosine Loss).
Нормализуем и W, и x: ||W||=1, ||x||=s. Margin вычитается из косинуса:
logit_y = s * (cos(theta_y) - m), для других классов logit = s * cos(theta).
Margin аддитивный в cos.

Концепция: ArcFace (Additive Angular Margin Loss).
Margin аддитивный в УГЛЕ: logit_y = s * cos(theta_y + m). Геометрически
красиво, на практике даёт лучший SOTA для face recognition.

Демо: вручную считаем logit'ы по этим формулам для одного примера.
"""
import numpy as np

rng = np.random.default_rng(0)

# === игрушечное: 3 класса, эмбеддинг x ===
np.set_printoptions(precision=3, suppress=True)
W = rng.normal(size=(3, 4))
W /= np.linalg.norm(W, axis=1, keepdims=True)  # нормализуем веса
x = rng.normal(size=4)
x_norm = x / np.linalg.norm(x)
y_true = 0
s = 30.0   # scale
m = 0.5    # margin

cos_theta = W @ x_norm   # косинусы углов между x и каждым W_c
theta = np.arccos(np.clip(cos_theta, -1, 1))
print(f"cos(theta) = {cos_theta}, theta = {theta}")

# === обычный softmax (по cosine) ===
logits_plain = s * cos_theta
p_plain = np.exp(logits_plain - logits_plain.max())
p_plain /= p_plain.sum()
print(f"Softmax probs:   {p_plain}")

# === CosFace: logit_y = s*(cos - m), остальные = s*cos ===
logits_cos = s * cos_theta.copy()
logits_cos[y_true] = s * (cos_theta[y_true] - m)
p_cos = np.exp(logits_cos - logits_cos.max())
p_cos /= p_cos.sum()
print(f"CosFace probs:   {p_cos}")

# === ArcFace: logit_y = s*cos(theta + m) ===
logits_arc = s * cos_theta.copy()
logits_arc[y_true] = s * np.cos(theta[y_true] + m)
p_arc = np.exp(logits_arc - logits_arc.max())
p_arc /= p_arc.sum()
print(f"ArcFace probs:   {p_arc}")

# === SphereFace (A-Softmax): logit_y = ||x|| * cos(m*theta) ===
m_int = 2
logits_sphere = np.linalg.norm(x) * cos_theta.copy()
logits_sphere[y_true] = np.linalg.norm(x) * np.cos(m_int * theta[y_true])
p_sphere = np.exp(logits_sphere - logits_sphere.max())
p_sphere /= p_sphere.sum()
print(f"SphereFace probs:{p_sphere}")

# === Center loss: тянем эмбеддинги к центру своего класса ===
centers = rng.normal(size=(3, 4))
emb = rng.normal(size=(20, 4))
labels = rng.integers(0, 3, size=20)
alpha = 0.1
for _ in range(200):
    for c in range(3):
        mask = labels == c
        if mask.any():
            centers[c] -= alpha * (centers[c] - emb[mask].mean(0))
L_center = 0.5 * np.sum((emb - centers[labels]) ** 2) / len(emb)
print(f"Center loss после обучения: {L_center:.3f}")

print("\nФормулы для запоминания:")
print("  CosFace:    logit_y = s * (cos(theta_y) - m)")
print("  ArcFace:    logit_y = s * cos(theta_y + m)")
print("  SphereFace: logit_y = ||x|| * cos(m * theta_y)")
print("  L-Softmax:  W_y^T x -> ||W_y|| ||x|| cos(m * theta_y)")
