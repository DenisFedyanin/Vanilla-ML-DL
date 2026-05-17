"""Раздел 37 — Внимание. Файл 1: варианты внимания.

Концепция: Additive (Bahdanau) attention.
score(q, k) = v^T tanh(W_q q + W_k k). Используется в RNN seq2seq.
v, W_q, W_k — обучаемые. Дорогое, но гибкое.

Концепция: Multiplicative (Luong) attention.
score(q, k) = q^T W k (general) или q^T k (dot). Дешевле additive.

Концепция: Scaled dot-product attention.
Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V.
Деление на sqrt(d_k) предотвращает большие значения softmax при больших d_k.
Базовая операция Transformer.

Концепция: матрица внимания.
Размер (T_q, T_k); строка i — распределение того, на какие ключи
смотрит запрос i. Сумма каждой строки = 1.
"""
import numpy as np

rng = np.random.default_rng(0)


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


# Маленький пример: T_q=2 запросов, T_k=4 ключа, d=3
T_q, T_k, d = 2, 4, 3
Q = rng.standard_normal((T_q, d))
K = rng.standard_normal((T_k, d))
V = rng.standard_normal((T_k, d))

# === Additive (Bahdanau) ===
print("=== Additive (Bahdanau) ===")
Wq = rng.standard_normal((d, d)) * 0.3
Wk = rng.standard_normal((d, d)) * 0.3
v = rng.standard_normal(d) * 0.3
scores_add = np.zeros((T_q, T_k))
for i in range(T_q):
    for j in range(T_k):
        scores_add[i, j] = v @ np.tanh(Q[i] @ Wq + K[j] @ Wk)
att_add = softmax(scores_add)
out_add = att_add @ V
print(f"scores{scores_add.shape}, attention\n{np.round(att_add, 3)}")
print(f"out_add{out_add.shape}, суммы строк attention = {att_add.sum(axis=1)}")

# === Multiplicative (Luong general) ===
print("\n=== Multiplicative (Luong) ===")
W_lu = rng.standard_normal((d, d)) * 0.3
scores_mul = Q @ W_lu @ K.T
att_mul = softmax(scores_mul)
out_mul = att_mul @ V
print(f"scores{scores_mul.shape}, attention\n{np.round(att_mul, 3)}")

# === Scaled dot-product ===
print("\n=== Scaled dot-product ===")
scores_sd = Q @ K.T / np.sqrt(d)
att_sd = softmax(scores_sd)
out_sd = att_sd @ V
print(f"QK^T/sqrt(d){scores_sd.shape}, attention\n{np.round(att_sd, 3)}")
print(f"out{out_sd.shape}")

# Проверим, как меняется softmax при росте d (без масштабирования он насыщается)
print("\nЭффект масштабирования:")
for dim in [4, 64, 512]:
    Qd = rng.standard_normal((1, dim))
    Kd = rng.standard_normal((4, dim))
    raw = (Qd @ Kd.T)[0]
    scaled = raw / np.sqrt(dim)
    print(f"d={dim}: |QK^T| range=[{raw.min():.2f},{raw.max():.2f}], scaled=[{scaled.min():.2f},{scaled.max():.2f}]")
