"""Раздел 53 — Recommenders. Файл 1: Базовые методы.

Концепция: Матрица user-item.
R[u, i] — оценка/факт взаимодействия. Обычно очень разреженная (>99% NaN).
Задача рекомендера — предсказать недостающие оценки или ранжировать товары.

Концепция: Popularity baseline.
Каждому пользователю рекомендуем самые популярные товары глобально.
Игнорирует индивидуальные предпочтения, но даёт нижнюю границу качества.

Концепция: Most-similar-user.
Находим ближайшего пользователя по cosine/Pearson на их векторе оценок,
рекомендуем то, что нравится ему, но не пробовал текущий.

Концепция: Most-similar-item.
По матрице R^T строим item-item сходства. Если пользователь оценил товар i,
рекомендуем самые похожие на i. Часто работает лучше user-based на больших
каталогах с малым числом активных пользователей.

Концепция: Cold start.
Новый пользователь / новый товар — нет истории. Popularity спасает для новых
пользователей; для новых товаров нужны контентные фичи.
"""
import numpy as np

rng = np.random.default_rng(0)

# Маленькая user-item матрица: 6 пользователей, 8 товаров, 0 = не оценено
R = np.array([
    [5, 4, 0, 0, 1, 0, 2, 0],
    [4, 5, 0, 0, 2, 0, 0, 1],
    [0, 0, 5, 4, 0, 4, 0, 0],
    [0, 1, 5, 5, 0, 4, 0, 0],
    [3, 0, 0, 0, 4, 0, 5, 4],
    [0, 2, 0, 1, 5, 0, 4, 5],
], dtype=float)
mask = R > 0


# === Popularity ===
pop = (R * mask).sum(axis=0) / np.maximum(mask.sum(axis=0), 1)
print(f"Средние оценки по товарам: {pop.round(2)}")
print(f"Top-3 популярных товара  : {np.argsort(-pop)[:3]}")


# === User-user similarity ===
def cosine_matrix(M):
    norm = np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
    Mn = M / norm
    return Mn @ Mn.T


U_sim = cosine_matrix(R)
np.fill_diagonal(U_sim, 0)
target_u = 0
nearest_u = int(np.argmax(U_sim[target_u]))
print(f"Пользователь 0 наиболее похож на пользователя {nearest_u} (sim={U_sim[target_u, nearest_u]:.2f})")
# Что рекомендуем 0? Товары, которые нравятся nearest_u, но не оценены у 0
unseen = ~mask[target_u]
candidates = R[nearest_u] * unseen
print(f"Рекомендуем пользователю 0 (most-similar-user): топ = {np.argsort(-candidates)[:3]}")


# === Item-item similarity ===
I_sim = cosine_matrix(R.T)
np.fill_diagonal(I_sim, 0)
# Для пользователя 0: оценки, взвешенные сходствами товаров
scores = np.zeros(R.shape[1])
for j in range(R.shape[1]):
    if mask[target_u, j]:
        continue
    sims = I_sim[j] * mask[target_u]
    rated = R[target_u] * mask[target_u]
    if sims.sum() > 0:
        scores[j] = (sims * rated).sum() / (np.abs(sims).sum() + 1e-9)
print(f"Item-item scores для пользователя 0: {scores.round(2)}")
print(f"Top-3 рекомендаций (item-item): {np.argsort(-scores)[:3]}")
