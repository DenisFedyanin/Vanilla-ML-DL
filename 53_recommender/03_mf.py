"""Раздел 53 — Recommenders. Файл 3: Matrix Factorization (MF).

Концепция: Идея MF.
Ищем R ≈ U V^T, где U ∈ R^{n_u × k}, V ∈ R^{n_i × k}. k латентных факторов
(жанры, темы). Каждый пользователь и каждый товар получает k-мерный вектор.

Концепция: Целевая функция.
L = sum_{(u,i) ∈ Observed} (R_{u,i} - U_u · V_i)^2 + lambda*(|U|^2 + |V|^2).
Считаем ошибку только по наблюдаемым ячейкам, остальные — пропуски.

Концепция: SGD-обучение.
e_{u,i} = R_{u,i} - U_u·V_i.
U_u <- U_u + lr * (e_{u,i} * V_i - lambda * U_u)
V_i <- V_i + lr * (e_{u,i} * U_u - lambda * V_i).
Делаем много проходов по случайным (u,i) с наблюдаемой R_{u,i}.

Концепция: Bias-члены.
Полная модель: R_{u,i} ≈ mu + b_u + b_i + U_u·V_i. mu — глобальное среднее,
b_u — смещение пользователя, b_i — смещение товара. Сильно улучшает качество.

Концепция: Применение и расширения.
Используется в Netflix-Prize классике. Расширения: ALS (попеременно фиксируем U
и V), implicit feedback (BPR), факторизационные машины (FM) для богатых фич.
"""
import numpy as np

rng = np.random.default_rng(0)

R = np.array([
    [5, 4, 0, 0, 1, 0, 2, 0],
    [4, 5, 0, 0, 2, 0, 0, 1],
    [0, 0, 5, 4, 0, 4, 0, 0],
    [0, 1, 5, 5, 0, 4, 0, 0],
    [3, 0, 0, 0, 4, 0, 5, 4],
    [0, 2, 0, 1, 5, 0, 4, 5],
], dtype=float)
mask = R > 0
n_u, n_i = R.shape
k = 3
lr = 0.02
lam = 0.05
mu = R[mask].mean()

U = rng.normal(scale=0.1, size=(n_u, k))
V = rng.normal(scale=0.1, size=(n_i, k))
bu = np.zeros(n_u)
bi = np.zeros(n_i)

pairs = [(u, i) for u in range(n_u) for i in range(n_i) if mask[u, i]]
for epoch in range(200):
    rng.shuffle(pairs)
    for u, i in pairs:
        pred = mu + bu[u] + bi[i] + U[u] @ V[i]
        e = R[u, i] - pred
        bu[u] += lr * (e - lam * bu[u])
        bi[i] += lr * (e - lam * bi[i])
        U_old = U[u].copy()
        U[u] += lr * (e * V[i] - lam * U[u])
        V[i] += lr * (e * U_old - lam * V[i])

R_hat = mu + bu[:, None] + bi[None, :] + U @ V.T
err = np.sqrt(((R - R_hat)[mask] ** 2).mean())
print(f"RMSE на наблюдаемых = {err:.3f}")
print(f"R (observed):\n{R}")
print(f"R_hat:\n{R_hat.round(2)}")
print(f"Заполнение пропусков для пользователя 0: {R_hat[0].round(2)}")
