"""Раздел 14 — Метрики расстояний.

Файл 1: расстояния между ВЕКТОРАМИ в R^d.

Концепция 177: Евклидово (L2).
d = sqrt(sum (x_i - y_i)^2). Самое привычное "по линейке". Чувствительно
к масштабу признаков => всегда нормализуйте перед использованием.

Концепция 178: Квадрат евклидова.
sum (x_i - y_i)^2. Часто используют вместо L2 (быстрее, без sqrt) когда
порядок не важен — в kNN, k-means и т.п.

Концепция 179: Manhattan (L1).
sum |x_i - y_i|. "Городские кварталы". Робастнее к выбросам, чем L2.

Концепция 180: Chebyshev (L_inf).
max |x_i - y_i|. Король на шахматной доске. Предел Minkowski-p при p->inf.

Концепция 181: Minkowski-p.
(sum |x_i - y_i|^p)^(1/p). Обобщение L1 (p=1), L2 (p=2), L_inf (p->inf).

Концепция 182: Canberra.
sum |x_i - y_i| / (|x_i| + |y_i|). Чувствительно к небольшим значениям
вблизи нуля.

Концепция 183: Bray-Curtis.
sum |x-y| / sum (x+y) для неотрицательных векторов. Часто в экологии для
сравнения видового состава.

Концепция 184: Cosine distance.
1 - cos_sim = 1 - (x·y)/(||x|| ||y||). Угол между векторами, безразмерно.
Стандарт для текстовых эмбеддингов.

Концепция 185: Pearson distance.
1 - корреляция Пирсона. Косинус после центрирования. Не зависит от среднего.
"""
import numpy as np

rng = np.random.default_rng(0)

x = np.array([1.0, 2.0, 3.0, 4.0])
y = np.array([2.0, 0.0, 2.0, 5.0])

# 177: Euclidean
d_eu = np.sqrt(((x - y) ** 2).sum())
print(f"177 Euclidean: {d_eu:.4f}")

# 178: Squared Euclidean
d_sq = ((x - y) ** 2).sum()
print(f"178 SqEuclidean: {d_sq:.4f}")

# 179: Manhattan
d_l1 = np.abs(x - y).sum()
print(f"179 Manhattan: {d_l1:.4f}")

# 180: Chebyshev
d_inf = np.abs(x - y).max()
print(f"180 Chebyshev: {d_inf:.4f}")

# 181: Minkowski p=3
p = 3
d_mk = (np.abs(x - y) ** p).sum() ** (1 / p)
print(f"181 Minkowski(p=3): {d_mk:.4f}")

# 182: Canberra
mask = (np.abs(x) + np.abs(y)) > 0
d_ca = (np.abs(x - y)[mask] / (np.abs(x) + np.abs(y))[mask]).sum()
print(f"182 Canberra: {d_ca:.4f}")

# 183: Bray-Curtis (на неотриц)
d_bc = np.abs(x - y).sum() / (x + y).sum()
print(f"183 Bray-Curtis: {d_bc:.4f}")

# 184: Cosine
cos_sim = (x @ y) / (np.linalg.norm(x) * np.linalg.norm(y))
print(f"184 Cosine dist: {1 - cos_sim:.4f} (sim={cos_sim:.4f})")

# 185: Pearson distance
xc, yc = x - x.mean(), y - y.mean()
corr = (xc @ yc) / (np.linalg.norm(xc) * np.linalg.norm(yc))
print(f"185 Pearson dist: {1 - corr:.4f} (corr={corr:.4f})")
