"""Раздел 21 — Расширенная кластеризация. Файл 1: варианты K-Means.

Концепция 621: Lloyd's K-Means.
Классический алгоритм: 1) приписать каждую точку к ближайшему центру;
2) пересчитать центр как среднее точек кластера; повторять до стабилизации.
Минимизирует сумму квадратов расстояний внутри кластеров (inertia).

Концепция 622: MiniBatch K-Means.
Каждый шаг центр обновляется по случайному батчу. Быстрее и допускает онлайн-режим,
качество чуть хуже Lloyd, но для больших данных это лучший выбор.

Концепция 623: K-Medoids (PAM — Partitioning Around Medoids).
Центр кластера — реальный объект (медоид), а не среднее. Устойчив к выбросам,
работает с произвольной метрикой расстояний.

Концепция 624: Bisecting K-Means.
Иерархический подход «top-down»: начинаем с одного кластера, на каждом шаге
делим самый «плохой» (max SSE) на 2 через обычный K-Means. Повторяем до K.
"""
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import MiniBatchKMeans, KMeans

rng = np.random.default_rng(0)

X, y_true = make_blobs(n_samples=300, centers=4, random_state=0, cluster_std=0.7)

# === 621: ручной Lloyd's ===
def lloyd_kmeans(X, K, n_iter=20, seed=0):
    r = np.random.default_rng(seed)
    idx = r.choice(len(X), K, replace=False)
    C = X[idx].copy()
    for _ in range(n_iter):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        labels = d.argmin(axis=1)
        new_C = np.array([X[labels == k].mean(axis=0)
                          if (labels == k).any() else C[k] for k in range(K)])
        if np.allclose(new_C, C):
            break
        C = new_C
    inertia = ((X - C[labels]) ** 2).sum()
    return labels, C, inertia

lab, C, inertia = lloyd_kmeans(X, K=4)
print(f"621 Lloyd K-Means: inertia={inertia:.1f}, центров={len(C)}")

# === 622: MiniBatch K-Means ===
mb = MiniBatchKMeans(n_clusters=4, batch_size=32, random_state=0, n_init=3).fit(X)
print(f"622 MiniBatch K-Means: inertia={mb.inertia_:.1f}")

# === 623: ручной K-Medoids (упрощённый PAM) ===
def kmedoids(X, K, n_iter=15, seed=0):
    r = np.random.default_rng(seed)
    med_idx = list(r.choice(len(X), K, replace=False))
    for _ in range(n_iter):
        D = np.sqrt(((X[:, None, :] - X[med_idx][None, :, :]) ** 2).sum(axis=2))
        labels = D.argmin(axis=1)
        new_med = []
        for k in range(K):
            members = np.where(labels == k)[0]
            if len(members) == 0:
                new_med.append(med_idx[k]); continue
            sub = X[members]
            dist_sum = np.sqrt(((sub[:, None, :] - sub[None, :, :]) ** 2).sum(2)).sum(axis=1)
            new_med.append(members[dist_sum.argmin()])
        if new_med == med_idx:
            break
        med_idx = new_med
    return labels, np.array(med_idx)

lab_med, med_idx = kmedoids(X, K=4)
print(f"623 K-Medoids: индексы медоидов={med_idx.tolist()}")

# === 624: Bisecting K-Means (простая реализация) ===
def bisecting_kmeans(X, K):
    clusters = [np.arange(len(X))]
    while len(clusters) < K:
        # самый «плохой» по SSE
        sses = [((X[idx] - X[idx].mean(0)) ** 2).sum() for idx in clusters]
        worst = int(np.argmax(sses))
        idx = clusters.pop(worst)
        km = KMeans(n_clusters=2, n_init=3, random_state=0).fit(X[idx])
        clusters.append(idx[km.labels_ == 0])
        clusters.append(idx[km.labels_ == 1])
    labels = np.empty(len(X), dtype=int)
    for k, idx in enumerate(clusters):
        labels[idx] = k
    return labels

lab_b = bisecting_kmeans(X, K=4)
print(f"624 Bisecting K-Means: размеры кластеров={np.bincount(lab_b).tolist()}")
