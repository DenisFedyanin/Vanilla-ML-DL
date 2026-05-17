"""Концепция 67: DBSCAN.

Кластеризация по плотности: точки в плотной области образуют кластер,
точки в редких областях считаются шумом (-1). Не нужно задавать число кластеров.
"""
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

X, _ = make_moons(n_samples=200, noise=0.05, random_state=0)
labels = DBSCAN(eps=0.2, min_samples=5).fit_predict(X)
print("Уникальные метки:", np.unique(labels))
print("Найдено кластеров:", (labels >= 0).any() and labels.max() + 1)
print("Точек-шумов:", (labels == -1).sum())
