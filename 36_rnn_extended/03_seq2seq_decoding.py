"""Раздел 36 — RNN. Файл 3: Seq2Seq и стратегии декодирования.

Концепция: Encoder-Decoder (Seq2Seq).
Encoder читает вход и сжимает в состояние; Decoder генерирует выход токен за токеном,
получая своё прошлое предсказание как следующий вход. Применение: перевод, summarization.

Концепция: Greedy decoding.
На каждом шаге берём argmax(p). Быстро, детерминированно, но может застрять
в локально хорошем, глобально плохом решении.

Концепция: Beam search.
Храним top-k гипотез. На каждом шаге расширяем все k путей словарём,
выбираем k лучших по суммарной log-вероятности. Лучше greedy.

Концепция: Length normalization.
Score = sum(log p) / length^α (α ~ 0.6-0.7). Без неё beam предпочитает короткие гипотезы.

Концепция: Top-k sampling.
Сэмплируем только из k наиболее вероятных слов. Балансирует качество и разнообразие.

Концепция: Top-p (nucleus) sampling.
Берём минимальное множество слов, у которого суммарная вероятность ≥ p.
Адаптивный размер словаря для каждого шага.
"""
import numpy as np

rng = np.random.default_rng(0)


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# Toy: словарь из 5 слов, длина 3
V = 5
T = 3
logits = rng.standard_normal((T, V))
probs = softmax(logits)
print(f"Распределения p_t (T={T}, V={V}):\n{np.round(probs, 2)}")

# === Greedy ===
greedy = np.argmax(probs, axis=-1)
print(f"\nGreedy: {greedy.tolist()}, log-prob = {sum(np.log(probs[t, greedy[t]]) for t in range(T)):.3f}")

# === Beam search (k=2), без модели — используем p_t из тех же probs ===
k = 2
beams = [(0.0, [])]    # (cum log-prob, tokens)
for t in range(T):
    cands = []
    for score, seq in beams:
        for v in range(V):
            cands.append((score + np.log(probs[t, v]), seq + [v]))
    cands.sort(key=lambda z: -z[0])
    beams = cands[:k]
print(f"\nBeam k={k}, лучшие гипотезы:")
for sc, seq in beams:
    print(f"  seq={seq}, score={sc:.3f}")

# === Length normalization ===
alpha = 0.7
print("\nС length normalization (alpha=0.7):")
for sc, seq in beams:
    norm_sc = sc / (len(seq) ** alpha)
    print(f"  seq={seq}, norm_score={norm_sc:.3f}")

# === Top-k sampling ===
p_last = probs[-1].copy()
top_k = 3
idx = np.argsort(-p_last)[:top_k]
p_topk = np.zeros_like(p_last)
p_topk[idx] = p_last[idx]
p_topk /= p_topk.sum()
sample = rng.choice(V, p=p_topk)
print(f"\nTop-k={top_k}: оставлены индексы {idx.tolist()}, sampled token={sample}")

# === Top-p (nucleus) sampling ===
p_thr = 0.7
sorted_idx = np.argsort(-p_last)
sorted_p = p_last[sorted_idx]
cum = np.cumsum(sorted_p)
cutoff = np.searchsorted(cum, p_thr) + 1
nucleus = sorted_idx[:cutoff]
p_nuc = np.zeros_like(p_last)
p_nuc[nucleus] = p_last[nucleus]
p_nuc /= p_nuc.sum()
sample = rng.choice(V, p=p_nuc)
print(f"Top-p={p_thr}: ядро из {cutoff} токенов {nucleus.tolist()}, sampled={sample}")
