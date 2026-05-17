"""Раздел 22 — Расширенное снижение размерности. Файл 3: Manifold learning.

Концепция 648: MDS — Multidimensional Scaling.
По матрице попарных расстояний строит низкоразмерное представление, в котором
расстояния сохраняются как можно лучше (стресс-функция).

Концепция 649: IsoMap.
Сначала строит граф ближайших соседей, считает «геодезические» расстояния
(кратчайший путь по графу) и применяет MDS. Хорошо разворачивает «свёрнутые»
поверхности (Swiss roll и т.п.).

Концепция 650: LLE — Locally Linear Embedding.
Каждая точка восстанавливается как линейная комбинация соседей. Эти же веса
переносятся в низкую размерность, где ищем согласованное вложение.
Сохраняет локальную структуру.
"""
import numpy as np
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding

rng = np.random.default_rng(0)
X, color = make_swiss_roll(n_samples=300, random_state=0)

# === 648: MDS ===
mds = MDS(n_components=2, n_init=2, max_iter=100,
          dissimilarity="euclidean", random_state=0,
          normalized_stress="auto").fit_transform(X)
print(f"648 MDS: embedding shape={mds.shape}, "
      f"спан x=[{mds[:,0].min():.2f},{mds[:,0].max():.2f}]")

# === 649: Isomap ===
iso = Isomap(n_neighbors=10, n_components=2).fit_transform(X)
print(f"649 IsoMap: embedding shape={iso.shape}")

# === 650: LLE ===
lle = LocallyLinearEmbedding(n_neighbors=10, n_components=2,
                             random_state=0).fit_transform(X)
print(f"650 LLE: embedding shape={lle.shape}")

# Сравним, насколько embedding сохраняет порядок по «color» (истинной координате).
def order_corr(emb):
    # корреляция первой координаты embedding с цветом ленты
    return abs(np.corrcoef(emb[:, 0], color)[0, 1])

print(f"648 MDS  |corr с истинным параметром|={order_corr(mds):.3f}")
print(f"649 IsoMap |corr|={order_corr(iso):.3f} (обычно выше — разворачивает рулон)")
print(f"650 LLE  |corr|={order_corr(lle):.3f}")
