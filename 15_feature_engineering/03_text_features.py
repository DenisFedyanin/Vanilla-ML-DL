"""Раздел 15 — Feature Engineering.

Файл 3: текстовые признаки.

Концепция 214: Tokenization.
Разбиение строки на лексемы. Простейшее — по пробелам с lower-case;
лучше regex \\w+. Точные правила сильно влияют на размер словаря.

Концепция 215: N-grams (word).
Объединение n подряд идущих токенов. Биграмы ловят локальный контекст:
"не хорошо" != "хорошо".

Концепция 216: Char-grams.
Подстроки длины n из символов. Робастны к опечаткам, работают для имён
собственных, морфологически богатых языков.

Концепция 217: Bag-of-Words (BoW).
Документ -> вектор частот слов. Игнорирует порядок, но это очень сильный
baseline для классификации.

Концепция 218: TF-IDF.
tf * idf, где idf = log(N / df). Понижает вес "общих" слов. Стандарт
для поиска и текстовой классификации.

Концепция 219: HashingVectorizer.
То же, что BoW, но хеширование вместо словаря. Подходит для огромных
корпусов и онлайн-обучения.
"""
import re

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

docs = [
    "Кот спит на коврике",
    "Собака лает на кота",
    "Кот и собака — друзья",
]


# 214: Tokenization
def tokenize(s):
    return re.findall(r"\w+", s.lower())


tokens = [tokenize(d) for d in docs]
print(f"214 tokens[0]={tokens[0]}")

# 215: word bigrams
def ngrams(toks, n):
    return [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]


print(f"215 bigrams[1]={ngrams(tokens[1], 2)}")

# 216: char-grams (n=3) первого документа
def char_ngrams(s, n):
    s = s.lower()
    return [s[i:i + n] for i in range(len(s) - n + 1)]


print(f"216 char-grams(n=3)[0][:8]={char_ngrams(docs[0], 3)[:8]}")

# 217: BoW (manual)
vocab = sorted({t for toks in tokens for t in toks})
v_idx = {w: i for i, w in enumerate(vocab)}
bow = np.zeros((len(docs), len(vocab)), dtype=int)
for i, toks in enumerate(tokens):
    for t in toks:
        bow[i, v_idx[t]] += 1
print(f"217 BoW shape={bow.shape}, vocab_size={len(vocab)}")

# 218: TF-IDF (manual)
tf = bow / bow.sum(axis=1, keepdims=True)
N = len(docs)
df = (bow > 0).sum(axis=0)
idf = np.log(N / df) + 1.0  # smooth
tfidf = tf * idf
print(f"218 TF-IDF top-3 слов в doc0: {[vocab[i] for i in np.argsort(-tfidf[0])[:3]]}")

# 219: HashingVectorizer (sklearn)
hv = HashingVectorizer(n_features=16, alternate_sign=False, norm=None)
H = hv.transform(docs).toarray()
print(f"219 HashingVectorizer shape={H.shape}, sum per doc={H.sum(axis=1).astype(int)}")
