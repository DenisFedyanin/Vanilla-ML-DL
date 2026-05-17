"""Раздел 51 — Graph ML. Файл 3: Эмбеддинги вершин.

Концепция: Зачем эмбеддинги вершин.
Нужно представить вершины как векторы в R^d, чтобы их можно было подавать в
классические ML-модели (классификация, link prediction, кластеризация).

Концепция: DeepWalk.
1) Из каждой вершины запускаем случайные блуждания длины L (несколько раз).
2) Получаем 'предложения' из идентификаторов вершин.
3) Учим word2vec (Skip-gram) — соседи по блужданию должны быть близки.

Концепция: node2vec — смещённые блуждания.
Параметры p (return) и q (in-out): при q>1 блуждание остаётся в окрестности
(BFS-like, гомофилия), при q<1 уходит вдаль (DFS-like, структурные роли).
p>1 не возвращаемся туда, откуда пришли.

Концепция: LINE / факторизация смежности.
Учим E и F такие, что sigmoid(E @ F^T) ≈ A. Эквивалентно матричной факторизации
с логистической функцией. Быстрее random walk методов на больших графах.

Концепция: Использование.
Полученные эмбеддинги — фичи для downstream-задач: классификация вершин,
link prediction (через сходство), кластеризация (k-means), визуализация (t-SNE).
"""
import numpy as np

rng = np.random.default_rng(0)

# Граф из двух 'сообществ'
A = np.zeros((8, 8))
for i, j in [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3),
             (4, 5), (4, 6), (5, 6), (5, 7), (6, 7),
             (3, 4)]:
    A[i, j] = A[j, i] = 1
neighbors = [np.where(A[i] == 1)[0] for i in range(8)]


def random_walk(start, length=10, p=1.0, q=1.0):
    walk = [start]
    for _ in range(length - 1):
        cur = walk[-1]
        nbrs = neighbors[cur]
        if len(nbrs) == 0:
            break
        if len(walk) < 2:
            walk.append(int(rng.choice(nbrs)))
            continue
        prev = walk[-2]
        # node2vec весовая схема
        weights = []
        for v in nbrs:
            if v == prev:
                weights.append(1.0 / p)
            elif A[prev, v] == 1:
                weights.append(1.0)
            else:
                weights.append(1.0 / q)
        weights = np.array(weights) / sum(weights)
        walk.append(int(rng.choice(nbrs, p=weights)))
    return walk


walks = [random_walk(v, length=10) for v in range(8) for _ in range(20)]
print(f"DeepWalk: пример блуждания из v0 = {walks[0]}")

# Тренируем embeddings (упрощённый skip-gram, отрицательный сэмплинг)
d = 4
E = rng.normal(scale=0.1, size=(8, d))
F = rng.normal(scale=0.1, size=(8, d))
lr = 0.05
window = 2
for epoch in range(30):
    for w in walks:
        for i, v in enumerate(w):
            for j in range(max(0, i - window), min(len(w), i + window + 1)):
                if i == j:
                    continue
                u = w[j]
                pos = 1 / (1 + np.exp(-E[v] @ F[u]))
                g = (1 - pos)
                E[v] += lr * g * F[u]
                F[u] += lr * g * E[v]
                neg = int(rng.integers(0, 8))
                pneg = 1 / (1 + np.exp(-E[v] @ F[neg]))
                E[v] -= lr * pneg * F[neg]
                F[neg] -= lr * pneg * E[v]

# Косинусное сходство между парами
def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


print(f"cos(v0, v2)  внутри сообщества = {cos(E[0], E[2]):+.2f}")
print(f"cos(v0, v6)  между сообществами= {cos(E[0], E[6]):+.2f}")

# LINE-подобное: факторизуем sigmoid(E F^T) ~ A
Ef = rng.normal(scale=0.1, size=(8, 3))
Ff = rng.normal(scale=0.1, size=(8, 3))
for _ in range(500):
    P = 1 / (1 + np.exp(-Ef @ Ff.T))
    G = P - A
    Ef -= 0.05 * G @ Ff / 8
    Ff -= 0.05 * G.T @ Ef / 8
print(f"LINE-factor recon error = {np.abs(P - A).mean():.3f}")
