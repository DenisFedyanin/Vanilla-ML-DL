"""Концепция 68: PCA - метод главных компонент.

Находим направления (оси), вдоль которых данные имеют наибольшую дисперсию,
и проектируем на эти оси. Сжатие данных с минимальной потерей информации.
"""
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 5))
X[:, 1] = X[:, 0] * 2 + 0.1 * rng.normal(size=100)   # столбец 1 коррелирует с 0

Xc = X - X.mean(0)
cov = Xc.T @ Xc / (len(X) - 1)
vals, vecs = np.linalg.eigh(cov)
order = np.argsort(-vals)
vals, vecs = vals[order], vecs[:, order]

explained = vals / vals.sum()
print("Объяснённая дисперсия:", np.round(explained, 3))
X_proj = Xc @ vecs[:, :2]
print("Размер после PCA:", X_proj.shape)
