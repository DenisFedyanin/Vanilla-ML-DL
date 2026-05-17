"""Раздел 40 — Эмбеддинги. Файл 2: GloVe и FastText.

Концепция: GloVe.
Вместо предсказания контекста — факторизация матрицы со-встречаемости X_ij.
Минимизируется: sum_{i,j} f(X_ij) (w_i · w_j + b_i + b_j - log X_ij)^2.
Веса f(x) подавляют редкие/слишком частые пары: f(x) = (x/x_max)^a при x<x_max.

Концепция: FastText (subword).
Вектор слова = сумма векторов n-грамм символов + само слово.
"where" с триграммами: <wh, whe, her, ere, re>.
Хорошо работает для OOV и морфологически богатых языков.
"""
import numpy as np

rng = np.random.default_rng(0)

# === Подготовим toy co-occurrence ===
corpus = "cat dog cat dog fish fish water water river river ocean ocean dog cat".split()
vocab = sorted(set(corpus))
w2i = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
window = 1

X = np.zeros((V, V))
for i, w in enumerate(corpus):
    for j in range(max(0, i - window), min(len(corpus), i + window + 1)):
        if i != j:
            X[w2i[w], w2i[corpus[j]]] += 1
print(f"vocab: {vocab}")
print(f"co-occurrence X:\n{X.astype(int)}")

# === GloVe SGD ===
d = 6
W = rng.standard_normal((V, d)) * 0.1
b = np.zeros(V)
x_max = 5.0
alpha = 0.75
lr = 0.05

def f(x): return min(1.0, (x / x_max) ** alpha)


for ep in range(200):
    total = 0.0
    for i in range(V):
        for j in range(V):
            xij = X[i, j]
            if xij == 0:
                continue
            wt = f(xij)
            diff = W[i] @ W[j] + b[i] + b[j] - np.log(xij)
            total += wt * diff ** 2
            g = 2 * wt * diff
            W[i], W[j] = W[i] - lr * g * W[j], W[j] - lr * g * W[i]
            b[i] -= lr * g
            b[j] -= lr * g

print(f"\nGloVe loss финал ≈ {total:.3f}")

def cos(a, b): return a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


print(f"cos(cat, dog)   = {cos(W[w2i['cat']], W[w2i['dog']]):.2f}")
print(f"cos(cat, water) = {cos(W[w2i['cat']], W[w2i['water']]):.2f}")

# === FastText subword n-grams ===
print("\n=== FastText subword ===")
def char_ngrams(w, n_min=3, n_max=4):
    w_pad = "<" + w + ">"
    out = []
    for n in range(n_min, n_max + 1):
        for i in range(len(w_pad) - n + 1):
            out.append(w_pad[i:i + n])
    out.append(w)
    return out


print(f"'where' -> {char_ngrams('where')[:8]}...")
# Tabula rasa: представляем слово как сумму векторов его n-грамм
all_grams = sorted({g for w in vocab for g in char_ngrams(w)})
g2i = {g: i for i, g in enumerate(all_grams)}
G = rng.standard_normal((len(all_grams), d)) * 0.1


def word_vec_ft(w):
    return sum(G[g2i[g]] for g in char_ngrams(w))


print(f"cos_ft(cat, dog)={cos(word_vec_ft('cat'), word_vec_ft('dog')):.2f}")
print("OOV слово 'cats' тоже получит вектор через свои n-граммы — преимущество FastText")
