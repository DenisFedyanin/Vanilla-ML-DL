"""Раздел 28 — Слои нейронных сетей. Файл 1: Dense, Embedding, Flatten.

Концепция 280: Linear / Dense layer.
Базовый слой полносвязной сети: y = x @ W + b.
x — вход (B, in), W — матрица весов (in, out), b — смещение (out,).
Производная по W = x.T @ grad, по x = grad @ W.T.
Используется почти везде: MLP, голова классификатора, FFN в трансформере.

Концепция 281: Bias.
Свободный член, позволяет сдвигать активацию. Без bias модель проходит через 0.
В батч-форме просто прибавляется broadcast-ом ко всем строкам.

Концепция 282: Embedding layer.
Таблица векторов размером (vocab, dim). По id-токена возвращает строку.
Это просто индексирование W[ids]; обучаемая lookup-таблица.
Применяется в NLP (слова->векторы), рекомендациях (user_id, item_id).

Концепция 283: Flatten / Reshape.
Превращает многомерный тензор (B, C, H, W) в (B, C*H*W).
Используется на стыке свёрток и Dense. Reshape — общая операция.
"""
import numpy as np

rng = np.random.default_rng(0)

# === 280: Linear / Dense ===
B, in_f, out_f = 4, 5, 3
x = rng.standard_normal((B, in_f))
W = rng.standard_normal((in_f, out_f)) * 0.1
b = np.zeros(out_f)
y = x @ W + b
print(f"280 Dense: x{ x.shape } @ W{ W.shape } -> y{ y.shape }")

# === 281: Bias ===
b2 = np.array([1.0, -1.0, 0.5])
y_no_bias = x @ W
y_bias = x @ W + b2
print(f"281 Bias сдвигает выход: средн без bias={y_no_bias.mean():.3f}, с bias={y_bias.mean():.3f}")

# === 282: Embedding ===
vocab, dim = 10, 4
E = rng.standard_normal((vocab, dim)) * 0.1
ids = np.array([0, 3, 7, 3])      # последовательность токенов
emb = E[ids]                       # просто индексация = forward embedding-слоя
print(f"282 Embedding: ids={ids.tolist()} -> emb{emb.shape}")
print(f"    одинаковые id дают одинаковый вектор: {np.allclose(emb[1], emb[3])}")

# === 283: Flatten / Reshape ===
img = rng.standard_normal((2, 3, 4, 4))    # B,C,H,W
flat = img.reshape(img.shape[0], -1)
print(f"283 Flatten: {img.shape} -> {flat.shape}")

# Полный мини-проход: ids -> Embed -> mean-pool -> Dense -> logits
ids = np.array([[1, 2, 3], [4, 5, 6]])
emb = E[ids]                                # (B, T, dim)
pooled = emb.mean(axis=1)                   # (B, dim) — mean over tokens
W2 = rng.standard_normal((dim, 3)) * 0.1
logits = pooled @ W2
print(f"Мини-MLP: ids{ids.shape} -> emb{emb.shape} -> pool{pooled.shape} -> logits{logits.shape}")
