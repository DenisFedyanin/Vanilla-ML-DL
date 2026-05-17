"""Раздел 20 — Семейство бустингов. Файл 1: AdaBoost и его варианты.

Концепция 600: AdaBoost — общая идея.
Последовательное обучение «слабых» классификаторов. Каждый следующий уделяет
больше внимания объектам, на которых предыдущие ошибались. Итоговое
предсказание — взвешенное голосование. Хорош для табличных данных.

Концепция 601: AdaBoost SAMME (Stagewise Additive Modeling).
Многоклассовая версия AdaBoost. Веса классификаторов:
alpha_t = log((1-err)/err) + log(K-1), где K — число классов.
При K=2 совпадает с классическим AdaBoost-M1.

Концепция 602: AdaBoost.M1.
Бинарная классификация. Веса объектов: w_i *= exp(alpha * I[y_i != h_t(x_i)]).
После каждого шага веса нормируются.

Концепция 603: AdaBoost.R2 (идея регрессии).
Каждому объекту назначается относительная ошибка |y - h(x)| / max_err.
Веса умножаются на beta^(1-err_i), где beta = avg_err/(1-avg_err).
Это AdaBoost для регрессии, в sklearn — AdaBoostRegressor.

Концепция 604: Слабые ученики — pen' (decision stumps).
Деревья глубины 1. Делают один порог по одному признаку. По одиночке слабые
(чуть лучше случайного), но в ансамбле — мощные.
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

rng = np.random.default_rng(0)

# === 600-602: Ручной AdaBoost-M1 со stumps ===
X, y = make_classification(n_samples=200, n_features=4, n_informative=3,
                           n_redundant=0, random_state=0)
y_s = np.where(y == 0, -1, 1)  # ±1 для удобства
n = len(y_s)
w = np.full(n, 1 / n)
stumps, alphas = [], []
T = 20
for t in range(T):
    stump = DecisionTreeClassifier(max_depth=1)
    stump.fit(X, y_s, sample_weight=w)
    pred = stump.predict(X)
    err = np.sum(w * (pred != y_s)) / np.sum(w)
    err = max(err, 1e-10)
    alpha = 0.5 * np.log((1 - err) / err)
    w = w * np.exp(-alpha * y_s * pred)
    w /= w.sum()
    stumps.append(stump)
    alphas.append(alpha)

scores = sum(a * s.predict(X) for a, s in zip(alphas, stumps))
acc_manual = (np.sign(scores) == y_s).mean()
print(f"600-602 Ручной AdaBoost-M1 (T={T} stumps): accuracy={acc_manual:.3f}")

# === 601: SAMME через sklearn ===
clf = AdaBoostClassifier(n_estimators=20, algorithm="SAMME", random_state=0)
clf.fit(X, y)
print(f"601 sklearn AdaBoost SAMME: accuracy={clf.score(X, y):.3f}")

# === 603: AdaBoost.R2 — концепт ===
# Идея: на каждой итерации обучаем регрессор, считаем относительную ошибку
# err_i = |y_i - h(x_i)| / max_err; beta = avg_err/(1-avg_err);
# веса w_i *= beta^(1 - err_i). Чем меньше ошибка — тем сильнее «успокаиваем» объект.
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import make_regression

Xr, yr = make_regression(n_samples=200, n_features=4, noise=5.0, random_state=0)
reg = AdaBoostRegressor(n_estimators=20, loss="linear", random_state=0)
reg.fit(Xr, yr)
print(f"603 AdaBoost.R2 (sklearn AdaBoostRegressor): R^2={reg.score(Xr, yr):.3f}")

# === 604: один stump — это базовый слабый ученик ===
single = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y)
print(f"604 Один stump в одиночку: accuracy={single.score(X, y):.3f} "
      f"(против ансамбля {acc_manual:.3f})")
