"""Раздел 21 — Расширенная кластеризация. Файл 4: Spectral, Affinity, BIRCH.

Концепция 633: Spectral clustering.
Строим граф сходства (W_ij = exp(-||x-y||^2/sigma)), считаем нормализованный
лапласиан L = I - D^{-1/2} W D^{-1/2}, берём первые k собственных векторов и
кластеризуем их K-Means. Хорошо ловит «не-выпуклые» формы.

Концепция 634: Affinity Propagation.
Точки обмениваются «сообщениями» (responsibility, availability), пока не
выберут «экземпляры» — реальные точки-центры. K не задаётся, определяется по
данным. Дорогой O(N^2).

Концепция 635: BIRCH.
Балансированное иерархическое дерево CF (clustering features) для онлайн-агрегации
большого объёма данных. Хорош для очень больших датасетов в памяти/потоке.
В конце можно «дозавершить» K-Means или Agglomerative.
"""
import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import SpectralClustering, AffinityPropagation, Birch

rng = np.random.default_rng(0)

# === 633: Spectral на «лунах» (не-выпуклая форма) ===
X_m, _ = make_moons(n_samples=200, noise=0.05, random_state=0)
spec = SpectralClustering(n_clusters=2, affinity="nearest_neighbors",
                          n_neighbors=10, random_state=0,
                          assign_labels="kmeans").fit(X_m)
print(f"633 Spectral на moons: размеры кластеров={np.bincount(spec.labels_).tolist()}")

# === 634: Affinity Propagation ===
X_b, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.6, random_state=0)
ap = AffinityPropagation(random_state=0, damping=0.9).fit(X_b)
print(f"634 AffinityPropagation: найдено exemplars={len(ap.cluster_centers_indices_)}, "
      f"кластеров={len(set(ap.labels_))}")

# === 635: BIRCH (онлайн-агрегация) ===
brc = Birch(n_clusters=3, threshold=0.5).fit(X_b)
print(f"635 BIRCH: размеры кластеров={np.bincount(brc.labels_).tolist()}")
