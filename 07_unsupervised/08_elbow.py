"""Концепция 71: Elbow method для выбора K в K-Means.

Считаем inertia (сумма квадратов расстояний до центров) для разных K.
Выбираем K в 'локте' графика - там, где падение замедляется.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=4, random_state=0)
for k in range(1, 8):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    print(f"k={k}: inertia={km.inertia_:.1f}")
print("Ищем 'локоть' - после k=4 падение становится медленным.")
