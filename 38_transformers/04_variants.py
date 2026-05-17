"""Раздел 38 — Трансформеры. Файл 4: weight tying, KV cache, parallel residual, MoE.

Концепция: Weight tying.
Embedding-таблицу (V, d) переиспользуют как матрицу выходного проектора (d, V) (транспонированную).
Экономит V*d параметров и часто улучшает PPL.

Концепция: KV cache (декодирование).
При авторегрессии K и V для уже сгенерированных токенов не меняются -> кешируем их
и на каждом шаге добавляем только K_t, V_t. Линейное по T декодирование.

Концепция: Parallel residual (GPT-J).
y = x + Attn(LN(x)) + FFN(LN(x)). Внимание и FFN считаются параллельно,
а не последовательно. Быстрее, качество близко.

Концепция: Mixture of Experts (MoE).
Несколько FFN-экспертов; router выбирает top-k для каждого токена.
Эффективная "ширина" большая, реальный compute — только активных экспертов.
"""
import numpy as np

rng = np.random.default_rng(0)


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


# === Weight tying ===
print("=== Weight tying ===")
V, d = 100, 16
E = rng.standard_normal((V, d)) * 0.02
ids = np.array([3, 7, 2])
h = E[ids]                                  # embedding lookup
logits = h @ E.T                            # tied output projection
print(f"E{E.shape}; ids{ids.shape} -> h{h.shape} -> logits{logits.shape}")
print(f"Сэкономлено V*d={V*d} параметров (не нужна отдельная W_out)")

# === KV cache ===
print("\n=== KV cache (autoregressive) ===")
d_k = 8
K_cache = np.zeros((0, d_k))
V_cache = np.zeros((0, d_k))
for t in range(4):
    # новый токен -> один Q, и его K_t, V_t
    Q_t = rng.standard_normal((1, d_k))
    K_t = rng.standard_normal((1, d_k))
    V_t = rng.standard_normal((1, d_k))
    K_cache = np.concatenate([K_cache, K_t], axis=0)
    V_cache = np.concatenate([V_cache, V_t], axis=0)
    scores = Q_t @ K_cache.T / np.sqrt(d_k)
    attn = softmax(scores, axis=-1)
    out = attn @ V_cache
    print(f"  step{t}: cache K{K_cache.shape}, attn{attn.shape}, out{out.shape}")
print("Без кеша мы бы пересчитывали K,V по всем токенам на каждом шаге.")

# === Parallel residual (GPT-J) ===
print("\n=== Parallel residual ===")
T = 3
x = rng.standard_normal((T, d))
def ln(x):
    mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + 1e-5)

# Имитируем Attn и FFN
W_a = rng.standard_normal((d, d)) * 0.1
W_f = rng.standard_normal((d, d)) * 0.1
attn_out = ln(x) @ W_a
ffn_out = np.maximum(0, ln(x) @ W_f)
y_parallel = x + attn_out + ffn_out
y_serial = x + attn_out
y_serial = y_serial + np.maximum(0, ln(y_serial) @ W_f)
print(f"parallel y{y_parallel.shape} (Attn и FFN из одного входа LN(x))")
print(f"serial y{y_serial.shape} (FFN видит уже изменённый residual)")

# === MoE: top-k gating ===
print("\n=== MoE (4 эксперта, top-2 gating) ===")
n_experts, top_k = 4, 2
T, d = 5, 6
x_moe = rng.standard_normal((T, d))
W_router = rng.standard_normal((d, n_experts)) * 0.3
gate_logits = x_moe @ W_router                          # (T, E)
gates = softmax(gate_logits, axis=-1)
# выбираем top-k экспертов для каждого токена
top_idx = np.argsort(-gates, axis=-1)[:, :top_k]
print(f"gate logits{gate_logits.shape}; top-{top_k} эксперты для каждого токена:\n{top_idx}")
print("Только выбранные эксперты обрабатывают токен — sparse compute")
