"""Раздел 38 — Трансформеры. Файл 2: Decoder block.

Концепция: Decoder block.
Три под-слоя:
  1) Masked self-attention (causal) — токен не видит будущее.
  2) Cross-attention — Q из decoder, K/V из encoder.
  3) FFN.
Каждый с Add & Norm.

Концепция: Masked self-attention.
Применяется треугольная маска -inf над диагональю, чтобы при обучении teacher forcing
позиция t не подсматривала t+1.

Концепция: Cross-attention (encoder-decoder).
Запросы из последовательности декодера; ключи и значения из выхода энкодера.
Позволяет декодеру выбирать релевантную информацию из источника (перевод и т.д.).
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


def attention(Q, K, V, mask=None):
    d = Q.shape[-1]
    s = Q @ K.T / np.sqrt(d)
    if mask is not None:
        s = s + mask
    return softmax(s, axis=-1) @ V


# Размеры
T_dec, T_enc, d = 4, 6, 8
x_dec = rng.standard_normal((T_dec, d))
enc_out = rng.standard_normal((T_enc, d))

# Параметры (упрощённо: single-head)
Wq1 = rng.standard_normal((d, d)) * 0.2
Wk1 = rng.standard_normal((d, d)) * 0.2
Wv1 = rng.standard_normal((d, d)) * 0.2

Wq2 = rng.standard_normal((d, d)) * 0.2
Wk2 = rng.standard_normal((d, d)) * 0.2
Wv2 = rng.standard_normal((d, d)) * 0.2

W1 = rng.standard_normal((d, 4 * d)) * 0.2
W2 = rng.standard_normal((4 * d, d)) * 0.2

print("=== Transformer Decoder block ===")
print(f"x_dec{x_dec.shape}, enc_out{enc_out.shape}")

# 1) Masked self-attention
causal = np.triu(np.full((T_dec, T_dec), -np.inf), k=1)
Q1, K1, V1 = x_dec @ Wq1, x_dec @ Wk1, x_dec @ Wv1
self_out = attention(Q1, K1, V1, mask=causal)
x1 = layer_norm(x_dec + self_out)
print(f"masked self-att out{self_out.shape} -> x1{x1.shape}")

# 2) Cross-attention
Q2 = x1 @ Wq2
K2 = enc_out @ Wk2
V2 = enc_out @ Wv2
cross_out = attention(Q2, K2, V2)
x2 = layer_norm(x1 + cross_out)
print(f"cross-att out{cross_out.shape} -> x2{x2.shape}")

# 3) FFN
ffn = np.maximum(0, x2 @ W1) @ W2
y = layer_norm(x2 + ffn)
print(f"ffn out -> y{y.shape}")

# Подтверждение причинности: attention для t=0 видит только t=0
s_check = x_dec @ Wq1 @ (x_dec @ Wk1).T / np.sqrt(d) + causal
a_check = softmax(s_check, axis=-1)
print(f"\nПервая строка self-attention (видит только себя):\n{np.round(a_check[0], 2)}")
print(f"Вторая строка (видит токены 0 и 1):\n{np.round(a_check[1], 2)}")
