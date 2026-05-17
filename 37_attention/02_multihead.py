"""Раздел 37 — Внимание. Файл 2: Multi-head attention.

Концепция: Multi-head attention.
Вместо одного внимания размера d_model — h голов размера d_k = d_model/h.
Каждая голова имеет свои проекции W_Q, W_K, W_V, и смотрит на разные подпространства.
Их выходы конкатенируются и проектируются W_O.

Концепция: split/concat по головам.
Тензор (B, T, d_model) -> (B, T, h, d_k) -> transpose -> (B, h, T, d_k).
После внимания обратно (B, T, h, d_k) -> (B, T, d_model).

Концепция: параметры MHA.
W_Q, W_K, W_V, W_O — каждая (d_model, d_model). Всего 4 * d_model^2 параметров.
"""
import numpy as np

rng = np.random.default_rng(0)


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


# Параметры
B, T, d_model, h = 1, 5, 8, 4
d_k = d_model // h
print(f"B={B}, T={T}, d_model={d_model}, heads={h}, d_k={d_k}")

# Вход
x = rng.standard_normal((B, T, d_model))

# Проекции
W_Q = rng.standard_normal((d_model, d_model)) * 0.2
W_K = rng.standard_normal((d_model, d_model)) * 0.2
W_V = rng.standard_normal((d_model, d_model)) * 0.2
W_O = rng.standard_normal((d_model, d_model)) * 0.2

Q = x @ W_Q
K = x @ W_K
V = x @ W_V
print(f"Q{Q.shape}, K{K.shape}, V{V.shape}")

# Split на головы: (B, T, d_model) -> (B, T, h, d_k) -> (B, h, T, d_k)
def split(t):
    B_, T_, _ = t.shape
    return t.reshape(B_, T_, h, d_k).transpose(0, 2, 1, 3)


Qh = split(Q)
Kh = split(K)
Vh = split(V)
print(f"После split: Qh{Qh.shape}")

# Scaled dot-product на каждой голове
scores = Qh @ Kh.transpose(0, 1, 3, 2) / np.sqrt(d_k)
attn = softmax(scores, axis=-1)
out_h = attn @ Vh
print(f"attn{attn.shape}, out_per_head{out_h.shape}")

# Concat: (B, h, T, d_k) -> (B, T, h*d_k) = (B, T, d_model)
out = out_h.transpose(0, 2, 1, 3).reshape(B, T, d_model)
print(f"После concat: out{out.shape}")

# Финальная проекция
y = out @ W_O
print(f"Итог MHA: y{y.shape}")
print(f"Параметры: 4 * d_model^2 = {4 * d_model * d_model}")

# Что каждая голова смотрит на разное:
print("\nAttention каждой головы для запроса t=0:")
for hd in range(h):
    a = attn[0, hd, 0]
    print(f"  head{hd}: {np.round(a, 2).tolist()}")
