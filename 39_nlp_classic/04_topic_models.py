"""Раздел 39 — NLP. Файл 4: тематические модели (LDA).

Концепция: LDA (Latent Dirichlet Allocation).
Генеративная модель: каждый документ — смесь тем, каждая тема — распределение по словам.
Темы и доли тем восстанавливают из bag-of-words через Variational Bayes или Gibbs.

Концепция: Dirichlet prior.
Распределение над симплексом; α<1 -> разреженные смеси (документ обычно про 1-2 темы).

Концепция: интерпретация.
Каждая тема — это набор топ-слов с наибольшей вероятностью.
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

docs = [
    "cat dog cat mouse pet",
    "dog cat cat pet dog",
    "river fish water ocean fish",
    "fish ocean water river",
    "python code data python",
    "code data python data code",
]

cv = CountVectorizer()
X = cv.fit_transform(docs)
vocab = cv.get_feature_names_out()
print(f"vocab: {vocab.tolist()}")
print(f"X{X.shape}")

# === Обучаем LDA с 3 темами ===
lda = LatentDirichletAllocation(n_components=3, max_iter=30, learning_method="batch", random_state=0)
doc_topic = lda.fit_transform(X)
topic_word = lda.components_ / lda.components_.sum(axis=1, keepdims=True)

print("\n=== Топ-слова каждой темы ===")
for t in range(3):
    top = np.argsort(-topic_word[t])[:4]
    words = [(vocab[i], round(topic_word[t, i], 2)) for i in top]
    print(f"  topic {t}: {words}")

print("\n=== Распределение тем по документам ===")
for d, dist in zip(docs, doc_topic):
    print(f"  '{d}' -> {np.round(dist, 2)}")
print("\nДокументы про животных/природу/код кластеризуются по разным темам.")
