"""Концепция 65: K-Means++ инициализация.

Чтобы алгоритм не зацикливался в плохом локальном оптимуме, первый центр
берётся случайно, а каждый следующий - с вероятностью ~ d(x)^2 (квадрату
расстояния до ближайшего уже выбранного центра).
"""
import numpy as np
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=4, random_state=0)
rng = np.random.default_rng(0)

centers = [X[rng.integers(len(X))]]
for _ in range(3):
    d2 = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
    probs = d2 / d2.sum()
    centers.append(X[rng.choice(len(X), p=probs)])

centers = np.array(centers)
print("K-Means++ начальные центры:\n", centers)
