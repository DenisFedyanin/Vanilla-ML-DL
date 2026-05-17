"""Раздел 39 — NLP. Файл 2: n-grams и Bag-of-Words.

Концепция: word n-grams.
Подпоследовательности из n подряд идущих слов. Сохраняют локальный порядок.
"the cat sat" -> 2-граммы: ("the","cat"), ("cat","sat").

Концепция: character n-grams.
То же, но по символам — устойчиво к опечаткам, работает на out-of-vocabulary.

Концепция: Bag-of-Words.
Текст -> вектор частот слов из общего словаря. Порядок теряется.
Простейший признаковый разбор для классификации.

Концепция: нормализация BoW.
Делим на длину документа (TF), либо L2-нормировка вектора.
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


docs = [
    "the cat sat on the mat",
    "the dog sat on the rug",
    "cats and dogs are pets",
]

# === Word n-grams (вручную) ===
print("=== Word 2-grams ===")
def word_ngrams(text, n):
    toks = text.split()
    return [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]


bigrams = word_ngrams(docs[0], 2)
print(f"'{docs[0]}' -> {bigrams}")

# === Char n-grams ===
print("\n=== Char 3-grams ===")
def char_ngrams(text, n):
    return [text[i:i + n] for i in range(len(text) - n + 1)]


cg = char_ngrams("hello", 3)
print(f"'hello' -> {cg}")

# === BoW (вручную) ===
print("\n=== Manual BoW ===")
vocab = sorted({w for d in docs for w in d.split()})
print(f"словарь ({len(vocab)} слов): {vocab}")

def bow(doc):
    counts = np.zeros(len(vocab), dtype=int)
    for w in doc.split():
        counts[vocab.index(w)] += 1
    return counts


X = np.stack([bow(d) for d in docs])
print(f"матрица BoW{X.shape}:\n{X}")

# === BoW через sklearn ===
print("\n=== sklearn CountVectorizer ===")
cv = CountVectorizer()
X_sk = cv.fit_transform(docs).toarray()
print(f"vocab sklearn: {cv.get_feature_names_out().tolist()}")
print(f"X_sk{X_sk.shape}:\n{X_sk}")

# === Нормализация: TF (делим на длину) и L2 ===
tf = X / X.sum(axis=1, keepdims=True)
print(f"\nTF (нормировка на длину):\n{np.round(tf, 2)}")
l2 = X / np.linalg.norm(X, axis=1, keepdims=True)
print(f"L2-нормировка:\n{np.round(l2, 2)}")
