"""Концепция 97: Word embeddings.

Каждому слову сопоставим вектор. Похожие слова - близкие векторы.
Здесь - просто look-up table (как embedding layer в нейросетях).
"""
import numpy as np

vocab = ["король", "королева", "мужчина", "женщина", "машина", "автомобиль"]
rng = np.random.default_rng(0)
emb = rng.normal(size=(len(vocab), 4))  # 4D эмбеддинги
# Имитируем 'осмысленные' эмбеддинги - близкие синонимам
emb[5] = emb[4] + 0.05 * rng.normal(size=4)
emb[1] = emb[0] - emb[2] + emb[3]


def cos(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


print(f"sim(король, королева)        = {cos(emb[0], emb[1]):.3f}")
print(f"sim(машина, автомобиль)      = {cos(emb[4], emb[5]):.3f}")
print(f"sim(король, машина)          = {cos(emb[0], emb[4]):.3f}")
print("Аналогия: король - мужчина + женщина ~ королева")
analogy = emb[0] - emb[2] + emb[3]
sims = [(w, cos(analogy, e)) for w, e in zip(vocab, emb)]
print(sorted(sims, key=lambda x: -x[1]))
