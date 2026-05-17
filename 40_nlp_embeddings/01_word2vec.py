"""Раздел 40 — Эмбеддинги слов. Файл 1: word2vec Skip-gram.

Концепция: Skip-gram.
Целевое слово предсказывает контекст. Модель: для каждого слова w пара эмбеддингов:
input v_w и output u_w. p(context|w) ∝ exp(u_c · v_w).
В отличие от CBOW, который усредняет контекст.

Концепция: Negative sampling.
Вместо softmax по всему словарю — бинарная классификация:
true pair (w, c+): хотим sigmoid(u_c+ · v_w) -> 1.
k случайных negative pairs (w, c-): sigmoid(u_c- · v_w) -> 0.

Концепция: косинусное сходство.
После обучения близкие слова имеют близкие вектора (cos > 0).
"""
import numpy as np

rng = np.random.default_rng(0)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


# === Toy corpus ===
corpus = "cat dog cat dog fish fish water water river river ocean ocean dog".split()
vocab = sorted(set(corpus))
w2i = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
print(f"vocab ({V}): {vocab}")

# === Сбор (target, context) пар, window=1 ===
pairs = []
for i, w in enumerate(corpus):
    for j in (i - 1, i + 1):
        if 0 <= j < len(corpus):
            pairs.append((w2i[w], w2i[corpus[j]]))
print(f"пар (target,context): {len(pairs)}")

# === Инициализация ===
d = 8
V_in = rng.standard_normal((V, d)) * 0.1            # input embeddings
V_out = rng.standard_normal((V, d)) * 0.1           # output embeddings

# === Обучение с negative sampling ===
lr = 0.1
n_epochs = 200
k_neg = 3
for epoch in range(n_epochs):
    rng.shuffle(pairs)
    loss_e = 0.0
    for tgt, ctx in pairs:
        v_t = V_in[tgt]
        u_c = V_out[ctx]
        # positive sample
        score = v_t @ u_c
        pred = sigmoid(score)
        grad_pos = pred - 1
        loss_e += -np.log(pred + 1e-9)
        V_in[tgt] -= lr * grad_pos * u_c
        V_out[ctx] -= lr * grad_pos * v_t
        # negative samples
        for _ in range(k_neg):
            neg = rng.integers(0, V)
            u_n = V_out[neg]
            sc = v_t @ u_n
            p_n = sigmoid(sc)
            grad_neg = p_n
            loss_e += -np.log(1 - p_n + 1e-9)
            V_in[tgt] -= lr * grad_neg * u_n
            V_out[neg] -= lr * grad_neg * v_t

print(f"loss финал ≈ {loss_e / len(pairs):.3f}")

# === Косинусные сходства ===
def cos(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


print("\nСходства (cos):")
for a in ["cat", "fish", "river"]:
    sims = sorted(((w, cos(V_in[w2i[a]], V_in[w2i[w]])) for w in vocab if w != a),
                  key=lambda z: -z[1])[:3]
    print(f"  {a}: {[(w, round(s, 2)) for w, s in sims]}")
