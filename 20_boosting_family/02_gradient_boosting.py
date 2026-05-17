"""Раздел 20 — Семейство бустингов. Файл 2: Gradient Boosting.

Концепция 605: Gradient Boosting — общая идея.
Бустинг как покоординатный градиентный спуск в пространстве функций.
F_{m}(x) = F_{m-1}(x) + nu * h_m(x), где h_m обучается на отрицательном градиенте
функции потерь по предсказаниям F_{m-1}.

Концепция 606: GB для регрессии (MSE) — обучение на остатках.
При MSE: grad = (F - y), значит -grad = y - F = «остаток».
То есть каждое следующее дерево предсказывает остаток предыдущей композиции.

Концепция 607: GB для классификации (log-loss) — концепт.
Для бинарной классификации с log-loss отрицательный градиент по logit:
r_i = y_i - sigmoid(F_{m-1}(x_i)). На таких «псевдо-остатках» учим следующий регрессор.
Итоговое предсказание: sigmoid(F_M(x)).

Концепция 608: Shrinkage (learning rate nu).
Малое nu (0.05-0.1) делает шаг осторожнее и обычно требует больше деревьев,
но даёт лучшее обобщение. Это сильнейший регуляризатор бустинга.

Концепция 609: n_estimators и переобучение.
Слишком много деревьев → переобучение на train. Подбирают по валидации или
ранней остановкой (early stopping).
"""
import numpy as np
from sklearn.datasets import make_regression, make_classification
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier

rng = np.random.default_rng(0)

# === 605-606: ручной GB-регрессор на остатках ===
X, y = make_regression(n_samples=200, n_features=3, noise=3.0, random_state=0)
F = np.full_like(y, y.mean(), dtype=float)  # начальное приближение
nu = 0.1
trees = []
for m in range(50):
    residual = y - F  # отрицательный градиент MSE
    t = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X, residual)
    F += nu * t.predict(X)
    trees.append(t)

mse_manual = np.mean((y - F) ** 2)
print(f"605-606 Ручной GB (50 деревьев, nu=0.1): train MSE={mse_manual:.2f}")

# === 607: GB-классификатор (log-loss) — концепт + sklearn ===
Xc, yc = make_classification(n_samples=200, n_features=4, random_state=0)
gbc = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1,
                                 max_depth=3, random_state=0)
gbc.fit(Xc, yc)
print(f"607 GB-classifier (log-loss) sklearn: accuracy={gbc.score(Xc, yc):.3f}")
# Псевдо-остатки на 1-м шаге: r = y - sigmoid(F0), где F0=log(p/(1-p)) от среднего.
p0 = yc.mean()
F0 = np.log(p0 / (1 - p0))
pseudo = yc - 1 / (1 + np.exp(-F0))
print(f"     pseudo-residual mean при F0=const: {pseudo.mean():.4f} (≈0)")

# === 608: эффект shrinkage ===
for nu_try in (0.5, 0.1, 0.02):
    Fk = np.full_like(y, y.mean(), dtype=float)
    for t in trees:
        Fk += nu_try * t.predict(X)
    mse_k = np.mean((y - Fk) ** 2)
    print(f"608 nu={nu_try}: train MSE={mse_k:.2f}")

# === 609: n_estimators ===
for M in (5, 20, 50):
    g = GradientBoostingClassifier(n_estimators=M, learning_rate=0.1,
                                   max_depth=3, random_state=0).fit(Xc, yc)
    print(f"609 n_estimators={M}: train acc={g.score(Xc, yc):.3f}")
