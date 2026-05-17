"""Концепция 34: Silhouette score (метрика качества кластеризации).

Для каждой точки: s = (b - a) / max(a, b),
где a - среднее расстояние до своего кластера, b - до ближайшего чужого.
Среднее по всем точкам, от -1 до 1. Чем выше - тем лучше.
"""
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=300, centers=3, random_state=0)
for k in [2, 3, 4, 5]:
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    print(f"k={k}: silhouette={silhouette_score(X, labels):.3f}")
