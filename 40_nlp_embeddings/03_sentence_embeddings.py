"""Раздел 40 — Эмбеддинги. Файл 3: представления предложений.

Концепция: mean/max pooling.
Простейший способ получить вектор предложения — усреднить или взять max
по векторам его слов. Используется как сильный baseline.

Концепция: SIF (Smooth Inverse Frequency).
Взвешиваем слова коэффициентом a/(a + p(w)), затем вычитаем главную компоненту.
Сильный baseline без обучения.

Концепция: Doc2Vec (PV-DM, PV-DBOW).
Расширение word2vec: добавляется обучаемый вектор документа,
который участвует в предсказании контекста наряду со словами.
"""
import numpy as np

rng = np.random.default_rng(0)

# Возьмём заранее заданные "эмбеддинги слов" (вместо реального обучения)
vocab = ["cat", "dog", "fish", "water", "river", "ocean", "code", "python", "data", "the"]
d = 8
emb = {w: rng.standard_normal(d) * 0.3 for w in vocab}
# Сделаем "cat" и "dog" похожими; "fish/water/river" — отдельный кластер; "code/python/data" — третий
emb["dog"] = emb["cat"] + rng.standard_normal(d) * 0.05
emb["fish"] = np.array([2.0] * d) + rng.standard_normal(d) * 0.05
emb["water"] = emb["fish"] + rng.standard_normal(d) * 0.05
emb["river"] = emb["fish"] + rng.standard_normal(d) * 0.05
emb["ocean"] = emb["fish"] + rng.standard_normal(d) * 0.05
emb["code"] = np.array([-2.0] * d) + rng.standard_normal(d) * 0.05
emb["python"] = emb["code"] + rng.standard_normal(d) * 0.05
emb["data"] = emb["code"] + rng.standard_normal(d) * 0.05

sentences = [
    "the cat dog",
    "fish water river ocean",
    "python code data",
    "cat dog fish",
]

# === Mean pooling ===
def mean_emb(sent):
    return np.mean([emb[w] for w in sent.split() if w in emb], axis=0)


# === Max pooling ===
def max_emb(sent):
    return np.max([emb[w] for w in sent.split() if w in emb], axis=0)


# === SIF: взвешивание IDF-like + удаление главной компоненты ===
freq = {"the": 0.5, "cat": 0.1, "dog": 0.1, "fish": 0.1, "water": 0.05,
        "river": 0.05, "ocean": 0.05, "code": 0.1, "python": 0.1, "data": 0.1}
a = 1e-3
def sif_emb(sent):
    toks = sent.split()
    vs = [a / (a + freq.get(w, 0.01)) * emb[w] for w in toks if w in emb]
    return np.mean(vs, axis=0)


def cos(x, y): return x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9)


print("=== Sentence embedding similarities ===")
for kind, fn in [("mean", mean_emb), ("max", max_emb), ("SIF", sif_emb)]:
    print(f"\n--- {kind} pooling ---")
    vecs = [fn(s) for s in sentences]
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            print(f"  '{sentences[i]}' vs '{sentences[j]}': cos={cos(vecs[i], vecs[j]):.2f}")

print("\n=== Doc2Vec (концепция) ===")
print("PV-DM: к контексту слов конкатенируем (или усредняем) вектор документа d_i,")
print("предсказываем следующее слово -> учим и word-векторы, и doc-векторы.")
print("PV-DBOW: документ предсказывает случайные слова из себя (как Skip-gram).")
