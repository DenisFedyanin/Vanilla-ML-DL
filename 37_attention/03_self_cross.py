"""Раздел 37 — Внимание. Файл 3: self-attention, cross-attention, маски.

Концепция: Self-attention.
Q, K, V вычисляются из ОДНОЙ и той же последовательности x.
Каждый токен смотрит на все остальные (включая себя).
Базовая операция в encoder Transformer.

Концепция: Cross-attention.
Q берётся из decoder, K и V — из выхода encoder.
Используется в seq2seq Transformer (перевод): decoder "запрашивает" encoder.

Концепция: Causal (look-ahead) mask.
В decoder для авторегрессии токен t не должен видеть t+1, t+2, ...
Маска — нижне-треугольная матрица; верхняя половина = -inf перед softmax.

Концепция: Padding mask.
Если последовательности в батче разной длины, добавляют PAD-токены.
Маска -inf для PAD-позиций, чтобы внимание их игнорировало.
"""
import numpy as np

rng = np.random.default_rng(0)


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dp(Q, K, V, mask=None):
    d = Q.shape[-1]
    s = Q @ K.T / np.sqrt(d)
    if mask is not None:
        s = s + mask
    a = softmax(s, axis=-1)
    return a @ V, a


# === Self-attention ===
print("=== Self-attention ===")
T, d = 4, 6
x = rng.standard_normal((T, d))
Wq = rng.standard_normal((d, d)) * 0.2
Wk = rng.standard_normal((d, d)) * 0.2
Wv = rng.standard_normal((d, d)) * 0.2
Q = x @ Wq
K = x @ Wk
V = x @ Wv
out, a = scaled_dp(Q, K, V)
print(f"x{x.shape}; self-att: out{out.shape}, attention\n{np.round(a, 2)}")

# === Cross-attention: Q из decoder, K/V из encoder ===
print("\n=== Cross-attention ===")
T_dec, T_enc = 3, 5
x_dec = rng.standard_normal((T_dec, d))
x_enc = rng.standard_normal((T_enc, d))
Q_dec = x_dec @ Wq
K_enc = x_enc @ Wk
V_enc = x_enc @ Wv
out_x, a_x = scaled_dp(Q_dec, K_enc, V_enc)
print(f"Q (decoder){Q_dec.shape}, K/V (encoder){K_enc.shape}")
print(f"attention (T_dec x T_enc){a_x.shape}: каждая строка decoder-запросит {T_enc} encoder-ключей")

# === Causal (треугольная) маска ===
print("\n=== Causal mask ===")
mask = np.triu(np.full((T, T), -np.inf), k=1)
print(f"mask:\n{mask}")
out_c, a_c = scaled_dp(Q, K, V, mask=mask)
print(f"attention с causal mask\n{np.round(a_c, 2)}")
print("Видно, что верх-правый угол = 0 (запрещено смотреть в будущее)")

# === Padding mask ===
print("\n=== Padding mask ===")
# скажем, токены 3 и 4 — паддинг
pad_pos = np.array([0, 0, 0, 1, 1])    # 1 = pad
T2 = 5
x2 = rng.standard_normal((T2, d))
Q2 = x2 @ Wq
K2 = x2 @ Wk
V2 = x2 @ Wv
pad_mask = np.where(pad_pos[None, :] == 1, -np.inf, 0.0)
print(f"pad_mask (применяется к каждой строке):\n{pad_mask}")
out_p, a_p = scaled_dp(Q2, K2, V2, mask=pad_mask)
print(f"attention с PAD mask\n{np.round(a_p, 2)}")
print("Столбцы PAD-позиций должны быть нулевыми после softmax")
