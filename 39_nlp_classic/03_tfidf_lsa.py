"""Раздел 39 — NLP. Файл 3: TF-IDF и LSA.

Концепция: TF (term frequency).
tf(t,d) = count(t,d) / len(d) или просто count. Меряет, насколько часто термин в документе.

Концепция: IDF (inverse document frequency).
idf(t) = log(N / df(t)), где df(t) — в скольких документах встречается t.
Редкие слова получают больший вес.

Концепция: TF-IDF.
tfidf(t,d) = tf(t,d) * idf(t). Снижает вес общих слов, поднимает редкие.
Базовый способ построить признаки для классификации/поиска текстов.

Концепция: LSA (Latent Semantic Analysis).
TruncatedSVD на матрице (документы x термины). Получаем "темы" — латентные направления.
Документы и слова можно сравнивать в малом k-мерном пространстве.
"""
import numpy as np
from sklearn.decomposition import TruncatedSVD

docs = [
    "cat dog mouse",
    "dog cat dog",
    "fish water river",
    "river ocean fish",
    "code python data",
]

# === Manual TF-IDF ===
vocab = sorted({w for d in docs for w in d.split()})
V = len(vocab)
N = len(docs)
idx = {w: i for i, w in enumerate(vocab)}

tf = np.zeros((N, V))
for d_i, d in enumerate(docs):
    for w in d.split():
        tf[d_i, idx[w]] += 1

df = (tf > 0).sum(axis=0)
idf = np.log(N / df)
tfidf = tf * idf
print("=== Manual TF-IDF ===")
print(f"vocab: {vocab}")
print(f"df: {df}")
print(f"idf: {np.round(idf, 2)}")
print(f"TF-IDF matrix{tfidf.shape}:\n{np.round(tfidf, 2)}")

# === LSA через TruncatedSVD ===
print("\n=== LSA (k=2 темы) ===")
svd = TruncatedSVD(n_components=2, random_state=0)
doc_emb = svd.fit_transform(tfidf)
print(f"doc embeddings{doc_emb.shape}:")
for d, e in zip(docs, doc_emb):
    print(f"  '{d}': {np.round(e, 2)}")

# слова в том же пространстве: components_.T дает координаты слов
word_emb = svd.components_.T          # (V, k)
print("\nКоординаты слов в латентных темах:")
for w, e in zip(vocab, word_emb):
    print(f"  {w:8s}: {np.round(e, 2)}")

# Сходство документов через LSA
def cos(a, b): return a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
print(f"\nLSA cosine(doc0, doc1) = {cos(doc_emb[0], doc_emb[1]):.2f}")
print(f"LSA cosine(doc0, doc4) = {cos(doc_emb[0], doc_emb[4]):.2f}")
