"""Раздел 21 — Расширенная кластеризация. Файл 5: Fuzzy C-Means.

Концепция 636: Fuzzy C-Means (FCM).
В отличие от K-Means, каждая точка имеет «мягкую» принадлежность ВСЕМ кластерам:
u_ik in [0,1], sum_k u_ik = 1. Параметр m > 1 (обычно m=2) — степень нечёткости.

Концепция 637: Обновление принадлежностей.
u_ik = 1 / sum_j ( ||x_i - c_k|| / ||x_i - c_j|| )^{2/(m-1)}.
Чем ближе точка к центру k и дальше от остальных — тем выше u_ik.

Концепция 638: Обновление центров.
c_k = sum_i (u_ik^m * x_i) / sum_i u_ik^m.
То есть центр — взвешенное среднее всех точек с весами u_ik^m.

Концепция 639: Целевая функция.
J = sum_i sum_k u_ik^m * ||x_i - c_k||^2 → min при ограничении sum_k u_ik = 1.
"""
import numpy as np
from sklearn.datasets import make_blobs

rng = np.random.default_rng(0)
X, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.7, random_state=0)
N, D = X.shape
K = 3
m = 2.0

# === 636: случайная инициализация мягких принадлежностей ===
U = rng.uniform(size=(N, K))
U /= U.sum(axis=1, keepdims=True)

# === 637-639: итерации FCM ===
J_prev = np.inf
for it in range(30):
    # 638: центры
    Um = U ** m
    C = (Um.T @ X) / Um.sum(axis=0)[:, None]
    # расстояния
    Dmat = np.sqrt(((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)) + 1e-12
    # 639: целевая функция
    J = (Um * Dmat ** 2).sum()
    # 637: обновление U
    ratio = Dmat[:, :, None] / Dmat[:, None, :]
    U = 1.0 / (ratio ** (2 / (m - 1))).sum(axis=2)
    if it in (0, 1, 5, 15, 29):
        print(f"636-639 FCM iter={it:>2}: J={J:.2f}")
    if abs(J_prev - J) < 1e-4:
        break
    J_prev = J

hard = U.argmax(axis=1)
print(f"636 FCM итог: hard-размеры={np.bincount(hard).tolist()}, "
      f"max U по строке (mean)={U.max(axis=1).mean():.2f}")
print(f"638 центры: {C.round(2).tolist()}")
