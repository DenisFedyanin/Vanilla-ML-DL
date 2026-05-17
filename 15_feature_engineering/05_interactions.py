"""Раздел 15 — Feature Engineering.

Файл 5: взаимодействия признаков.

Концепция 226: Полиномиальные признаки.
Добавляем степени и произведения признаков до степени d. Для d=2 на
[a, b]: [a, b, a^2, a*b, b^2]. Сильно расширяет пространство и линейный
классификатор внезапно становится мощным (ценой переобучения).

Концепция 227: Попарные произведения.
Только cross-произведения a_i * a_j (без степеней). Часто полезно для
факторизованных моделей и линейной регрессии.

Концепция 228: Отношения.
a/b — нормировка одного признака другим (доход на члена семьи, площадь
на комнату). Бывает информативнее, чем исходные.

Концепция 229: Group-by агрегации.
Например, средний доход в районе, как новый признак для каждой строки.
Хороший способ "влить" информацию из других строк/категорий.
"""
import numpy as np

rng = np.random.default_rng(0)

X = rng.normal(size=(6, 3))  # 6 объектов, 3 признака
group = np.array(["A", "B", "A", "B", "A", "C"])
target = np.array([1.0, 2.0, 1.5, 2.5, 1.7, 3.0])

# 226: polynomial features (manual) до степени 2
def poly_features(X, degree=2):
    n, d = X.shape
    cols = [np.ones(n)]
    for i in range(d):
        cols.append(X[:, i])
    if degree >= 2:
        for i in range(d):
            for j in range(i, d):
                cols.append(X[:, i] * X[:, j])
    return np.column_stack(cols)


P = poly_features(X, degree=2)
print(f"226 poly(d=2) shape: {X.shape} -> {P.shape}")

# 227: pairwise products только
def pairwise(X):
    n, d = X.shape
    cols = []
    for i in range(d):
        for j in range(i + 1, d):
            cols.append(X[:, i] * X[:, j])
    return np.column_stack(cols)


print(f"227 pairwise shape: {pairwise(X).shape} (=C(3,2)=3 столбца)")

# 228: ratios — например, признак 0 / признак 1
eps = 1e-6
ratio = X[:, 0] / (X[:, 1] + eps)
print(f"228 ratio col0/col1: {ratio.round(3)}")

# 229: group-by aggregate (mean target per group)
group_mean = {}
for g in np.unique(group):
    group_mean[g] = target[group == g].mean()
agg = np.array([group_mean[g] for g in group])
print(f"229 group_mean_target: {agg.round(3)} (по группам {group.tolist()})")
