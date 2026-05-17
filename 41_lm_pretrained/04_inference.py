"""Раздел 41 — Предобученные LM. Файл 4: стратегии генерации.

Концепция: Greedy decoding.
На каждом шаге argmax(p). Детерминированно, но скучно и часто повторяет.

Концепция: Sampling с температурой T.
p'_i ∝ p_i^(1/T). T<1 — резче (ближе к argmax), T>1 — более равномерно.

Концепция: Top-k sampling.
Сэмпл только из k самых вероятных токенов. Контролирует "хвост" распределения.

Концепция: Top-p (nucleus) sampling.
Берём минимум токенов, чьи кумулятивные вероятности ≥ p.
Адаптивный размер словаря — лучше top-k при разных формах распределения.

Концепция: repetition penalty.
Делим логиты уже встретившихся токенов на penalty>1. Уменьшает повторение.
"""
import numpy as np

rng = np.random.default_rng(0)


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


V = 10
logits = rng.standard_normal(V) * 2
print(f"исходные logits: {np.round(logits, 2)}")
probs = softmax(logits)
print(f"softmax: {np.round(probs, 3)}")

# === Greedy ===
print(f"\nGreedy: argmax = {int(np.argmax(probs))}")

# === Temperature ===
print("\n=== Temperature ===")
for T in [0.5, 1.0, 2.0]:
    p_T = softmax(logits / T)
    print(f"T={T}: top1_prob={p_T.max():.2f}, entropy={-(p_T * np.log(p_T + 1e-9)).sum():.2f}")

# === Top-k ===
print("\n=== Top-k (k=3) ===")
k = 3
top_idx = np.argsort(-probs)[:k]
p_k = np.zeros_like(probs)
p_k[top_idx] = probs[top_idx]
p_k /= p_k.sum()
print(f"kept indices: {top_idx.tolist()}, normalized probs: {np.round(p_k, 2)}")
print(f"sampled: {int(rng.choice(V, p=p_k))}")

# === Top-p (nucleus) ===
print("\n=== Top-p (p=0.8) ===")
p_thr = 0.8
sorted_idx = np.argsort(-probs)
sorted_p = probs[sorted_idx]
cum = np.cumsum(sorted_p)
cut = int(np.searchsorted(cum, p_thr)) + 1
nucleus = sorted_idx[:cut]
p_n = np.zeros_like(probs)
p_n[nucleus] = probs[nucleus]
p_n /= p_n.sum()
print(f"nucleus size={cut}, indices={nucleus.tolist()}")
print(f"sampled: {int(rng.choice(V, p=p_n))}")

# === Repetition penalty ===
print("\n=== Repetition penalty ===")
generated = [2, 5, 5, 2]
penalty = 1.5
log_pen = logits.copy()
for tok in set(generated):
    if log_pen[tok] > 0:
        log_pen[tok] /= penalty
    else:
        log_pen[tok] *= penalty
p_pen = softmax(log_pen)
print(f"уже сгенерированы: {generated}")
print(f"logits  before: {np.round(logits, 2)}")
print(f"logits  after : {np.round(log_pen, 2)}")
print(f"топ-токен (penalty): {int(np.argmax(p_pen))}")
