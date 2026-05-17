"""Раздел 23 — Матричные разложения. Файл 2: MF для коллаборативной фильтрации.

Концепция 659: Collaborative Filtering через матричное разложение.
Имеем матрицу рейтингов R (users x items), большая часть пропущена. Ищем
R ≈ P * Q^T, где P (U x k) — векторы пользователей, Q (I x k) — векторы фильмов.
Предсказание для (u,i): r_hat = P_u . Q_i.

Концепция 660: SGD-обновления Funk-SVD.
Только по известным (u,i): err = r - P_u.Q_i;
P_u <- P_u + lr * (err * Q_i - reg * P_u);
Q_i <- Q_i + lr * (err * P_u - reg * Q_i).

Концепция 661: ALS — Alternating Least Squares.
Поочерёдно фиксируем P и решаем по Q аналитически как линейную регрессию
(и наоборот). Каждый шаг — выпуклая задача, сходится быстро. Реализуем
упрощённый ALS-шаг.

Концепция 662: Регуляризация L2 на P и Q.
Без неё MF переобучается на «звёздных» пользователях/предметах. Член
+reg * (||P||^2 + ||Q||^2) добавляется в функцию потерь.
"""
import numpy as np

rng = np.random.default_rng(0)

# === 659: данные — маленькая user-item матрица с пропусками ===
R = np.array([
    [5, 3, 0, 1, 0],
    [4, 0, 0, 1, 2],
    [1, 1, 0, 5, 0],
    [0, 0, 5, 4, 0],
    [0, 1, 5, 4, 0],
], dtype=float)
mask = (R > 0).astype(float)
U_n, I_n = R.shape
k = 2
P = rng.normal(scale=0.1, size=(U_n, k))
Q = rng.normal(scale=0.1, size=(I_n, k))

# === 660: SGD Funk-SVD ===
lr, reg = 0.02, 0.05
for epoch in range(200):
    for u in range(U_n):
        for i in range(I_n):
            if mask[u, i] == 0:
                continue
            err = R[u, i] - P[u] @ Q[i]
            P[u] += lr * (err * Q[i] - reg * P[u])
            Q[i] += lr * (err * P[u] - reg * Q[i])

pred = P @ Q.T
rmse_sgd = np.sqrt(((R - pred) ** 2 * mask).sum() / mask.sum())
print(f"659-660 SGD MF: RMSE на известных={rmse_sgd:.3f}")
print(f"     предсказания для пропусков (округлены):\n{np.where(mask==0, pred, 0).round(2)}")

# === 661-662: ALS-шаги (один-два) ===
P2 = rng.normal(scale=0.1, size=(U_n, k))
Q2 = rng.normal(scale=0.1, size=(I_n, k))
reg_als = 0.1
for it in range(15):
    # фиксируем Q, решаем для каждого u: P_u = (Q^T W_u Q + reg I)^-1 Q^T W_u r_u
    for u in range(U_n):
        idx = np.where(mask[u] > 0)[0]
        if len(idx) == 0:
            continue
        Qi = Q2[idx]
        A = Qi.T @ Qi + reg_als * np.eye(k)
        b = Qi.T @ R[u, idx]
        P2[u] = np.linalg.solve(A, b)
    for i in range(I_n):
        idx = np.where(mask[:, i] > 0)[0]
        if len(idx) == 0:
            continue
        Pu = P2[idx]
        A = Pu.T @ Pu + reg_als * np.eye(k)
        b = Pu.T @ R[idx, i]
        Q2[i] = np.linalg.solve(A, b)

pred2 = P2 @ Q2.T
rmse_als = np.sqrt(((R - pred2) ** 2 * mask).sum() / mask.sum())
print(f"661-662 ALS MF: RMSE на известных={rmse_als:.3f}")
