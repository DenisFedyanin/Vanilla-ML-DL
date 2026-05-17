"""Раздел 37 — Внимание. Файл 4: эффективные варианты внимания.

Концепция: Sliding-window attention (Longformer).
Каждый токен смотрит только на w соседей слева/справа. Сложность O(T*w) вместо O(T^2).
+ небольшое глобальное внимание для специальных токенов (CLS).

Концепция: Sparse attention (Sparse Transformer, BigBird).
Заранее фиксированный шаблон ненулевых позиций в матрице внимания.
Комбинация window + random + global. Сохраняет долгий контекст за O(T sqrt T).

Концепция: Linformer / Performer — линейное внимание.
Linformer проецирует K, V в низкоразмерное пространство: T -> k << T.
Performer: использует random feature maps, чтобы аппроксимировать softmax(QK^T)
как ψ(Q) ψ(K)^T, давая линейную сложность.

Концепция: FlashAttention.
Не уменьшает FLOPs, а минимизирует чтение/запись в HBM памяти GPU.
Tiling: считаем софтмакс по блокам, поддерживая running max и sum (logsumexp trick).
Скорость x2-x4 при тех же результатах.
"""
import numpy as np

rng = np.random.default_rng(0)


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


# === Sliding-window mask ===
print("=== Sliding-window attention ===")
T, w = 8, 2
mask = np.full((T, T), -np.inf)
for i in range(T):
    lo, hi = max(0, i - w), min(T, i + w + 1)
    mask[i, lo:hi] = 0.0
print(f"window={w}; маска (0 разрешено, -inf запрещено):")
print(np.where(np.isinf(mask), -1, 0))

# Подсчёт ненулевых позиций
nnz = (mask == 0).sum()
print(f"Активных пар: {nnz} вместо T*T={T*T} (sparsity={nnz/(T*T):.2f})")

# === Sparse attention pattern: window + global ===
print("\n=== Sparse attention (window + global tokens) ===")
mask2 = np.full((T, T), -np.inf)
globals_ = [0]   # CLS-токен
for i in range(T):
    lo, hi = max(0, i - 1), min(T, i + 2)
    mask2[i, lo:hi] = 0.0
    for g in globals_:
        mask2[i, g] = 0.0
        mask2[g, i] = 0.0
print(f"Активных пар: {(mask2 == 0).sum()} ({len(globals_)} глобальных токенов)")

# === Linformer: проекция K, V вниз ===
print("\n=== Linformer (k_proj=4 vs T=16) ===")
T_l, d, k_proj = 16, 8, 4
K = rng.standard_normal((T_l, d))
V = rng.standard_normal((T_l, d))
E = rng.standard_normal((k_proj, T_l)) * 0.1
F = rng.standard_normal((k_proj, T_l)) * 0.1
K_low = E @ K     # (k_proj, d)
V_low = F @ V     # (k_proj, d)
print(f"K{K.shape} -> K_low{K_low.shape}; T sequence ужалась до k_proj")
# Внимание теперь O(T * k_proj) вместо O(T^2)
Q = rng.standard_normal((T_l, d))
attn = softmax(Q @ K_low.T / np.sqrt(d))
out = attn @ V_low
print(f"out{out.shape}, размер attention{attn.shape} (T x k_proj)")

# === Performer (linear attention) идея ===
print("\n=== Performer linear: softmax(QK)V ≈ ψ(Q)(ψ(K)^T V) ===")
# Простейшая ψ(x) = relu(x) (необучаемая, для иллюстрации)
psiQ = np.maximum(0, Q) + 1e-3
psiK = np.maximum(0, K) + 1e-3
KV = psiK.T @ V                  # (d, d) — не зависит от T_q
out_perf = psiQ @ KV             # (T, d)
print(f"ψ(Q){psiQ.shape}, KV{KV.shape}, out_perf{out_perf.shape} — O(T) сложность")

# === FlashAttention concept ===
print("\nFlashAttention: считаем softmax-блоками с running max и sum (logsumexp).")
print("Результат численно эквивалентен обычному, но не строит матрицу T x T в HBM.")
