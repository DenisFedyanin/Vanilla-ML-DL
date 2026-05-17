"""Раздел 53 — Recommenders. Файл 2: Collaborative Filtering.

Концепция: User-user CF.
Предсказание оценки r_{u,i} = baseline_u + sum_{v ∈ N(u)} sim(u,v) * (r_{v,i} - baseline_v)
                                            / sum |sim(u,v)|.
Усредняем оценки 'похожих' пользователей с учётом их собственных смещений.

Концепция: Item-item CF.
r_{u,i} = sum_{j ∈ N(i)} sim(i,j) * r_{u,j} / sum |sim(i,j)|.
Похожие товары имеют близкие оценки у одних и тех же людей. Часто стабильнее,
чем user-user, и проще обновлять (товары меняются медленнее, чем пользователи).

Концепция: Меры сходства.
Cosine: cos(u, v) = u·v / (|u||v|). Pearson: то же на центрированных данных
(вычитаем среднее пользователя). Pearson лучше учитывает разные шкалы оценок.

Концепция: Уход в среднее.
Нужно вычитать средние (per-user или per-item), чтобы 'строгие' и 'добрые'
пользователи не вносили систематический сдвиг.

Концепция: Sparse-tricks.
В реальности R разреженная. Считаем только по пересечению оценённых элементов.
Top-K соседей вместо всех, чтобы убрать шум и ускорить.
"""
import numpy as np

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

# Центрированные оценки (per user mean)
user_mean = np.array([R[u, mask[u]].mean() if mask[u].any() else 0 for u in range(n_u)])
Rc = np.where(mask, R - user_mean[:, None], 0)


def cos_sim_rows(M, masks):
    """Cosine sim по только пересечению оценённых компонентов."""
    n = M.shape[0]
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            common = masks[i] & masks[j]
            if common.sum() < 2:
                continue
            a = M[i, common]; b = M[j, common]
            d = np.linalg.norm(a) * np.linalg.norm(b) + 1e-9
            S[i, j] = (a @ b) / d
    return S


# === User-user CF ===
U_sim = cos_sim_rows(Rc, mask)
np.fill_diagonal(U_sim, 0)


def predict_uu(u, i, k=3):
    sims = U_sim[u].copy()
    sims[~mask[:, i]] = 0      # оставляем только тех, кто оценил i
    top = np.argsort(-np.abs(sims))[:k]
    num = sum(sims[v] * (R[v, i] - user_mean[v]) for v in top)
    den = sum(abs(sims[v]) for v in top) + 1e-9
    return user_mean[u] + num / den


print("User-user CF предсказания:")
for u, i in [(0, 2), (0, 5), (2, 0), (4, 1)]:
    print(f"  R[{u},{i}] ≈ {predict_uu(u, i):.2f}")


# === Item-item CF ===
I_sim = cos_sim_rows(Rc.T, mask.T)
np.fill_diagonal(I_sim, 0)


def predict_ii(u, i, k=3):
    sims = I_sim[i].copy()
    sims[~mask[u]] = 0
    top = np.argsort(-np.abs(sims))[:k]
    num = sum(sims[j] * R[u, j] for j in top)
    den = sum(abs(sims[j]) for j in top) + 1e-9
    return num / den if den > 1e-6 else user_mean[u]


print("Item-item CF предсказания:")
for u, i in [(0, 2), (0, 5), (2, 0), (4, 1)]:
    print(f"  R[{u},{i}] ≈ {predict_ii(u, i):.2f}")
