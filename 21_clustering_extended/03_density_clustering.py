"""Раздел 21 — Расширенная кластеризация. Файл 3: плотностные методы.

Концепция 629: Mean-Shift.
Каждая точка двигается в сторону среднего соседей в окне (bandwidth h).
В точках сходимости — моды плотности; кластер — все точки, пришедшие в одну моду.
Не требует K, но дорог по памяти и плохо масштабируется.

Концепция 630: DBSCAN (recap).
Плотностной алгоритм: core-точка имеет >= min_samples соседей в радиусе eps.
Точки соединяются через core → формируется кластер. Точки вне любого кластера —
шум. Не требует K, находит произвольные формы.

Концепция 631: OPTICS.
Обобщение DBSCAN, выдаёт «упорядочивание» точек по достижимости (reachability),
из которого можно выделить кластеры при разных eps. Лучше работает на данных
с переменной плотностью.

Концепция 632: HDBSCAN — концепт.
Hierarchical DBSCAN: строит дерево кластеров по плотности и выбирает «стабильные»
кластеры. В sklearn появился как `cluster.HDBSCAN` начиная с 1.3; иначе используем
DBSCAN с разными eps как заменитель.
"""
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift, DBSCAN, OPTICS

rng = np.random.default_rng(0)
X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.7, random_state=0)

# === 629: Mean-Shift ===
ms = MeanShift(bandwidth=1.0).fit(X)
print(f"629 Mean-Shift: найдено кластеров={len(set(ms.labels_))}, "
      f"размеры={np.bincount(ms.labels_).tolist()}")

# === 630: DBSCAN ===
db = DBSCAN(eps=0.6, min_samples=5).fit(X)
n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
n_noise = (db.labels_ == -1).sum()
print(f"630 DBSCAN(eps=0.6): кластеров={n_clusters}, шум={n_noise}")

# === 631: OPTICS ===
opt = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.1).fit(X)
n_opt = len(set(opt.labels_)) - (1 if -1 in opt.labels_ else 0)
print(f"631 OPTICS: кластеров={n_opt}, шум={(opt.labels_ == -1).sum()}")

# === 632: HDBSCAN-заменитель ===
try:
    from sklearn.cluster import HDBSCAN
    h = HDBSCAN(min_cluster_size=10).fit(X)
    print(f"632 HDBSCAN sklearn: кластеров="
          f"{len(set(h.labels_)) - (1 if -1 in h.labels_ else 0)}")
except ImportError:
    # Фоллбек: DBSCAN с разными eps — простая «иерархия».
    results = []
    for eps in (0.4, 0.6, 0.9, 1.5):
        d = DBSCAN(eps=eps, min_samples=5).fit(X)
        nc = len(set(d.labels_)) - (1 if -1 in d.labels_ else 0)
        results.append((eps, nc))
    print(f"632 HDBSCAN недоступен, прокси через DBSCAN(eps): {results}")
