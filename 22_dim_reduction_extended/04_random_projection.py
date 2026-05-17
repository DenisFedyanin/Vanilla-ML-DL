"""Раздел 22 — Расширенное снижение размерности. Файл 4: случайные проекции.

Концепция 651: Random Projection — идея.
Проектируем X (N x d) на случайную матрицу R (d x k), получаем X_red = X @ R.
Удивительно: попарные расстояния сохраняются с малой относительной ошибкой,
если k достаточно большое.

Концепция 652: Gaussian Random Projection.
R_ij ~ N(0, 1/k). Тогда E[||x R - y R||^2] = ||x - y||^2 — несмещённая оценка
квадратов расстояний. Просто реализуется вручную.

Концепция 653: Sparse Random Projection.
R_ij ∈ {-sqrt(s)/sqrt(k), 0, +sqrt(s)/sqrt(k)} с вероятностями 1/(2s), 1-1/s, 1/(2s).
Очень разреженная, считается быстрее, точность сравнима.

Концепция 654: Johnson-Lindenstrauss lemma.
Для N точек и допуска eps достаточно k = O(log(N) / eps^2), чтобы все попарные
расстояния сохранились с относительной ошибкой не более eps (с большой вероятностью).
Размерность d не входит — отсюда практическая сила.
"""
import numpy as np
from sklearn.random_projection import SparseRandomProjection, johnson_lindenstrauss_min_dim

rng = np.random.default_rng(0)
N, d = 200, 500
X = rng.normal(size=(N, d))

# === 651-652: Gaussian RP вручную ===
k = 100
R = rng.normal(0, 1 / np.sqrt(k), size=(d, k))
X_red = X @ R

# Сравним попарные расстояния (квадраты)
def pairwise_sq(M):
    return ((M[:, None, :] - M[None, :, :]) ** 2).sum(axis=2)

orig = pairwise_sq(X)
red = pairwise_sq(X_red)
mask = np.triu(np.ones_like(orig, dtype=bool), k=1)
rel_err = np.abs(red[mask] - orig[mask]) / (orig[mask] + 1e-12)
print(f"651-652 Gaussian RP d={d}->k={k}: средняя отн. ошибка попарных квадратов "
      f"расстояний={rel_err.mean():.3f}, max={rel_err.max():.3f}")

# === 653: Sparse RP через sklearn ===
srp = SparseRandomProjection(n_components=k, random_state=0).fit_transform(X)
red_sp = pairwise_sq(srp)
rel_err_sp = np.abs(red_sp[mask] - orig[mask]) / (orig[mask] + 1e-12)
print(f"653 Sparse RP: средняя отн. ошибка={rel_err_sp.mean():.3f}")

# === 654: формула Johnson-Lindenstrauss ===
for eps in (0.5, 0.3, 0.1):
    k_min = johnson_lindenstrauss_min_dim(n_samples=N, eps=eps)
    print(f"654 JL: для N={N}, eps={eps} нужно k >= {k_min}")
