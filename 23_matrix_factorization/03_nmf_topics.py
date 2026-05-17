"""Раздел 23 — Матричные разложения. Файл 3: NMF для тематического моделирования.

Концепция 663: NMF на TF-IDF матрице.
X (docs x terms) ≈ W (docs x topics) * H (topics x terms). Все элементы
неотрицательны, что даёт интерпретируемые «темы» — наборы слов с большими весами.

Концепция 664: Топ-термины для темы.
По строке H[t] выбираем индексы с самыми большими значениями — это и есть
ключевые слова темы t.

Концепция 665: Документ как смесь тем.
Строка W[d] показывает, какие темы присутствуют в документе d. Можно
интерпретировать как «доли тем» в документе (после нормировки).
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

docs = [
    "the cat sat on the mat",
    "dogs and cats are pets",
    "the dog barked at the cat",
    "neural networks learn from data",
    "deep learning uses neural networks",
    "machine learning models data patterns",
    "i love my dog and my cat",
    "convolutional neural networks for images",
]
vec = TfidfVectorizer(stop_words="english")
X = vec.fit_transform(docs)
terms = vec.get_feature_names_out()
print(f"663 TF-IDF матрица: shape={X.shape}, число термов={len(terms)}")

n_topics = 2
nmf = NMF(n_components=n_topics, init="nndsvd", max_iter=300, random_state=0)
W = nmf.fit_transform(X)
H = nmf.components_

# === 664: топ-3 термина для каждой темы ===
for t in range(n_topics):
    top_idx = np.argsort(-H[t])[:3]
    top_words = [terms[i] for i in top_idx]
    print(f"664 Тема {t}: ключевые слова = {top_words}")

# === 665: доли тем в первых 4 документах ===
W_norm = W / (W.sum(axis=1, keepdims=True) + 1e-12)
for d in range(4):
    print(f"665 doc[{d}] (\"{docs[d][:30]}...\"): "
          f"доли тем={W_norm[d].round(2).tolist()}")
