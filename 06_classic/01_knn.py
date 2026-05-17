"""Концепция 49: K-Nearest Neighbors.

Чтобы предсказать класс точки, смотрим на K её ближайших соседей в обучающей
выборке и берём наиболее частый класс.
"""
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(60, 2))
y = (X[:, 0] + X[:, 1] > 0).astype(int)


def knn_predict(X_train, y_train, x, k=5):
    d = np.linalg.norm(X_train - x, axis=1)
    nn = np.argsort(d)[:k]
    vals, counts = np.unique(y_train[nn], return_counts=True)
    return vals[counts.argmax()]


point = np.array([0.5, 0.3])
print("Точка:", point, "-> класс:", knn_predict(X, y, point, k=5))
