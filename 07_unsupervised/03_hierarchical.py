"""Концепция 66: Иерархическая (агломеративная) кластеризация.

Начинаем с того, что каждая точка - отдельный кластер. На каждом шаге
объединяем два ближайших, пока не останется нужное число.
"""
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=30, centers=3, random_state=0)
labels = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(X)
print("Метки кластеров:", labels)
