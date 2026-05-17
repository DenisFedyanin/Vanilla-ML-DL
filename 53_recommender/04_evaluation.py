"""Раздел 53 — Recommenders. Файл 4: Метрики качества.

Концепция: Precision@K.
Доля релевантных среди топ-K рекомендаций: |relevant ∩ topK| / K.
Простая и понятная. Не учитывает порядок внутри top-K.

Концепция: Recall@K.
Доля найденных релевантных от всех релевантных: |relevant ∩ topK| / |relevant|.
Важно, когда нужно 'не пропустить'.

Концепция: MAP@K (Mean Average Precision).
AP@K = (1/min(K, |rel|)) * sum_{j: rec_j релевантен} P@j.
Учитывает позицию — релевантный на 1-м месте важнее, чем на 10-м. MAP = среднее AP по пользователям.

Концепция: NDCG@K.
DCG = sum_{j=1..K} rel_j / log2(j + 1).  IDCG — идеальный порядок.
NDCG = DCG / IDCG ∈ [0, 1]. Поддерживает градированную релевантность.

Концепция: MRR (Mean Reciprocal Rank).
1 / позиция первого релевантного. Подходит для задач 'найти хоть один правильный'
(вопрос-ответ, поиск).

Концепция: Coverage и Diversity.
Coverage — какую долю каталога рекомендеры показывают пользователям.
Diversity — среднее попарное расстояние внутри топ-K. Помогает бороться с
'эффектом фильтра-пузыря', когда всем рекомендуется одно и то же.
"""
import numpy as np


def precision_at_k(rec, rel, k):
    rec_k = rec[:k]
    return sum(1 for r in rec_k if r in rel) / k


def recall_at_k(rec, rel, k):
    if not rel:
        return 0.0
    rec_k = rec[:k]
    return sum(1 for r in rec_k if r in rel) / len(rel)


def ap_at_k(rec, rel, k):
    score = 0.0
    hits = 0
    for j, r in enumerate(rec[:k], start=1):
        if r in rel:
            hits += 1
            score += hits / j
    return score / min(k, len(rel)) if rel else 0.0


def dcg_at_k(rec, rel, k):
    return sum((1 if r in rel else 0) / np.log2(j + 1) for j, r in enumerate(rec[:k], start=1))


def ndcg_at_k(rec, rel, k):
    ideal = dcg_at_k(list(rel)[:k], rel, k)
    return dcg_at_k(rec, rel, k) / ideal if ideal > 0 else 0.0


def mrr(rec, rel):
    for j, r in enumerate(rec, start=1):
        if r in rel:
            return 1.0 / j
    return 0.0


# Пример: 3 пользователя, каталог 10 товаров
users = {
    "u1": {"rec": [3, 1, 4, 5, 9, 2, 7], "rel": {1, 4, 7}},
    "u2": {"rec": [2, 6, 8, 0, 3, 5, 9], "rel": {6, 5}},
    "u3": {"rec": [0, 1, 2, 3, 4, 5, 6], "rel": {9}},
}
K = 5
catalog = set(range(10))

for u, d in users.items():
    print(f"{u}: P@{K}={precision_at_k(d['rec'], d['rel'], K):.2f}  "
          f"R@{K}={recall_at_k(d['rec'], d['rel'], K):.2f}  "
          f"AP@{K}={ap_at_k(d['rec'], d['rel'], K):.2f}  "
          f"NDCG@{K}={ndcg_at_k(d['rec'], d['rel'], K):.2f}  "
          f"MRR={mrr(d['rec'], d['rel']):.2f}")

map_score = np.mean([ap_at_k(d['rec'], d['rel'], K) for d in users.values()])
print(f"MAP@{K} = {map_score:.3f}")

# Coverage: какая доля каталога показана
shown = set().union(*[set(d['rec'][:K]) for d in users.values()])
print(f"Coverage@{K} = {len(shown) / len(catalog):.2f}")

# Diversity: среднее попарное Хэмминг-расстояние между рекомендациями пользователей
recs = [set(d['rec'][:K]) for d in users.values()]
pairs = [(i, j) for i in range(len(recs)) for j in range(i + 1, len(recs))]
div = np.mean([1 - len(recs[i] & recs[j]) / len(recs[i] | recs[j]) for i, j in pairs])
print(f"Diversity (Jaccard-distance между списками) = {div:.2f}")
