"""Раздел 38 — Трансформеры. Файл 1: Encoder block.

Концепция: Transformer encoder block (Vaswani, 2017).
Структура: x -> MultiHeadAttention -> Add&Norm -> FFN -> Add&Norm.
FFN: Linear(d, 4d) -> activation -> Linear(4d, d).
Residual соединения + LayerNorm стабилизируют обучение глубоких сетей.

Концепция: LayerNorm.
Нормировка по последней оси: (x - mean) / sqrt(var + eps), затем scale и shift.

Концепция: FFN (Feed-Forward Network).
Точечно-обновляет каждый токен (одинаково для всех T).
Расширение в 4 раза стандартно (в GPT, BERT).

Концепция: Add (residual).
Перевыходы добавляются ко входу: y = x + sublayer(x).
Помогает градиентам и сохраняет информацию.
"""
import numpy as np

rng = np.random.default_rng(0)


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def layer_norm(x, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def mha(x, Wq, Wk, Wv, Wo, h):
    T, d = x.shape
    d_k = d // h
    Q = (x @ Wq).reshape(T, h, d_k).transpose(1, 0, 2)   # (h, T, d_k)
    K = (x @ Wk).reshape(T, h, d_k).transpose(1, 0, 2)
    V = (x @ Wv).reshape(T, h, d_k).transpose(1, 0, 2)
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
    attn = softmax(scores, axis=-1)
    out = attn @ V                                       # (h, T, d_k)
    out = out.transpose(1, 0, 2).reshape(T, d)
    return out @ Wo


# === Encoder block forward ===
T, d, h = 5, 8, 2
x = rng.standard_normal((T, d))

# параметры MHA
Wq = rng.standard_normal((d, d)) * 0.2
Wk = rng.standard_normal((d, d)) * 0.2
Wv = rng.standard_normal((d, d)) * 0.2
Wo = rng.standard_normal((d, d)) * 0.2

# параметры FFN
W1 = rng.standard_normal((d, 4 * d)) * 0.2
W2 = rng.standard_normal((4 * d, d)) * 0.2

print("=== Transformer Encoder block ===")
print(f"input x{x.shape}")

# Sub-layer 1: MHA + residual + LN
mh = mha(x, Wq, Wk, Wv, Wo, h)
x1 = layer_norm(x + mh)
print(f"после MHA + Add&Norm: x1{x1.shape}")

# Sub-layer 2: FFN + residual + LN
ffn = gelu(x1 @ W1) @ W2
x2 = layer_norm(x1 + ffn)
print(f"после FFN + Add&Norm: x2{x2.shape}")

# Печать норм по шагам
print(f"\n||x||={np.linalg.norm(x):.2f} -> ||x1||={np.linalg.norm(x1):.2f} -> ||x2||={np.linalg.norm(x2):.2f}")
print("LayerNorm возвращает норму на стабильный уровень")
print(f"Параметры блока (без emb): 4*d^2 (MHA) + 2*4*d^2 (FFN) = {4*d*d + 2*4*d*d}")
