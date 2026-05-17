"""Концепция 64: K-Means.

Итеративно: 1) приписать каждую точку к ближайшему центру,
            2) пересчитать центры как среднее точек кластера.
"""
import numpy as np
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=3, random_state=0)
rng = np.random.default_rng(0)
k = 3
centers = X[rng.choice(len(X), k, replace=False)]

for it in range(20):
    d = np.linalg.norm(X[:, None] - centers[None], axis=2)
    labels = d.argmin(axis=1)
    new_centers = np.array([X[labels == j].mean(0) if (labels == j).any()
                            else centers[j] for j in range(k)])
    if np.allclose(new_centers, centers):
        print(f"Сошёлся на итерации {it}")
        break
    centers = new_centers

print("Центры:\n", centers)
