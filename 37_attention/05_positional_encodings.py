"""Раздел 37 — Внимание. Файл 5: positional encodings.

Концепция: Sinusoidal positional encoding (Vaswani, 2017).
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
Не обучается; даёт уникальный, гладкий векторный код для каждой позиции.
Позволяет обобщать на длины, не виденные при обучении.

Концепция: Learned positional embedding.
nn.Embedding(max_len, d). Обучается под данные. Проще, но ограничена max_len.

Концепция: Relative positional encoding (Shaw, T5).
Кодируем относительное расстояние i-j, а не абсолютное pos.
Лучше для длинных контекстов и работает с разными длинами.

Концепция: RoPE (Rotary Position Embedding).
Применяем 2D-поворот в плоскости каждой пары компонент q/k на угол, зависящий от pos.
q_rot = R(pos) q. Скалярное произведение q_rot(i) · k_rot(j) зависит только от i-j.

Концепция: ALiBi (Attention with Linear Biases).
Не модифицирует Q/K, а просто добавляет линейный bias -m*|i-j| к scores.
Бесплатно даёт хорошее обобщение на длинные тексты.
"""
import numpy as np

# === 1) Sinusoidal PE ===
print("=== Sinusoidal PE ===")
max_len, d = 8, 8
pos = np.arange(max_len)[:, None]                       # (T, 1)
i = np.arange(d)[None, :]                               # (1, d)
angle = pos / (10000 ** (2 * (i // 2) / d))
pe = np.where(i % 2 == 0, np.sin(angle), np.cos(angle))
print(f"PE{pe.shape}, первые две позиции:\n{np.round(pe[:2], 2)}")
print(f"||pe[t]|| ≈ const: {np.linalg.norm(pe, axis=1).round(2)}")

# === 2) Learned PE — просто nn.Embedding ===
print("\n=== Learned PE ===")
rng = np.random.default_rng(0)
emb = rng.standard_normal((max_len, d)) * 0.02
print(f"Таблица learned PE{emb.shape} — обучается как обычные веса")

# === 3) Relative PE ===
print("\n=== Relative PE ===")
# bias B[i,j] зависит только от i-j
B = np.zeros((max_len, max_len))
W_rel = rng.standard_normal(2 * max_len - 1) * 0.1
for i_ in range(max_len):
    for j_ in range(max_len):
        B[i_, j_] = W_rel[i_ - j_ + max_len - 1]
print(f"Relative bias{B.shape}, диагональ постоянна: {np.unique(np.diagonal(B)).size == 1}")

# === 4) RoPE (rotary) — 2D rotation на парах компонент ===
print("\n=== RoPE (rotary) ===")
def rope_apply(x, pos):
    """Поворот пар компонент на угол pos / 10000^(2i/d)."""
    d_x = x.shape[-1]
    half = d_x // 2
    freqs = 1.0 / (10000 ** (np.arange(half) / half))
    theta = pos * freqs                                 # (half,)
    cos = np.cos(theta)
    sin = np.sin(theta)
    x1 = x[..., :half]
    x2 = x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


q = rng.standard_normal(8)
k = rng.standard_normal(8)
# свойство: dot(rope(q, i), rope(k, j)) зависит только от i-j
val_5_2 = rope_apply(q, 5) @ rope_apply(k, 2)
val_8_5 = rope_apply(q, 8) @ rope_apply(k, 5)
print(f"<rope(q,5), rope(k,2)>={val_5_2:.3f}; разность позиций 3")
print(f"<rope(q,8), rope(k,5)>={val_8_5:.3f}; разность позиций 3 -> близко: {np.isclose(val_5_2, val_8_5, atol=1e-9)}")

# === 5) ALiBi: линейный bias по |i-j| ===
print("\n=== ALiBi ===")
T = 6
m = 0.25
i_, j_ = np.meshgrid(np.arange(T), np.arange(T), indexing="ij")
alibi = -m * np.abs(i_ - j_)
print(f"ALiBi bias (m={m}):\n{alibi}")
print("Добавляется напрямую к scores QK^T перед softmax")
