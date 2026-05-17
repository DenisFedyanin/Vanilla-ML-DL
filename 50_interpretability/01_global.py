"""Раздел 50 — Interpretability. Файл 1: Глобальная важность признаков.

Концепция: Глобальная vs локальная интерпретация.
Глобальная — какие фичи важны в среднем для модели на всём датасете.
Локальная — почему модель так предсказала конкретный объект.

Концепция: Коэффициенты линейной модели.
Для стандартизованных фич |w_j| напрямую отражает силу влияния. Знак — направление.
Не работает при коллинеарности (вес 'перетекает' между скоррелированными фичами).

Концепция: feature_importances_ в Random Forest.
Усреднённое уменьшение примеси (impurity, Gini/MSE), вызванное сплитами по фиче.
Быстро, но смещено в сторону фич с большим количеством уникальных значений.

Концепция: Permutation importance.
Перемешиваем значения одной фичи в test и смотрим, насколько упал скор.
Модель-агностична, не зависит от внутренностей. Минус — медленно при многих фичах.

Концепция: Когда они расходятся.
Линейные коэффициенты и RF importance часто противоречат друг другу — это сигнал
о нелинейности или коллинеарности. Permutation importance обычно надёжнее.
"""
import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)
X, y, true_w = make_regression(n_samples=400, n_features=6, n_informative=3,
                               coef=True, noise=5.0, random_state=0)
X = StandardScaler().fit_transform(X)
print(f"Истинные веса (informative — крупные): {true_w.round(1)}")

# Linear coefficients
lr = Ridge(alpha=1.0).fit(X, y)
print(f"Ridge |w|     : {np.abs(lr.coef_).round(2)}")

# RF importances
rf = RandomForestRegressor(n_estimators=80, random_state=0).fit(X, y)
print(f"RF importance : {rf.feature_importances_.round(3)}")


def permutation_importance(model, X, y, n_repeats=5):
    base = model.score(X, y)
    out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            drops.append(base - model.score(Xp, y))
        out[j] = np.mean(drops)
    return out


pi_lr = permutation_importance(lr, X, y)
pi_rf = permutation_importance(rf, X, y)
print(f"Permut.(Ridge): {pi_lr.round(3)}")
print(f"Permut.(RF)   : {pi_rf.round(3)}")
print("Топ-фичи у всех методов должны совпасть с informative из истинных w.")
