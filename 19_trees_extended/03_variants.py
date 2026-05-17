"""Раздел 19 — Деревья (расширенно).

Файл 3: варианты деревьев.

Концепция 282: ExtraTrees vs RandomForest.
RF: bootstrap-выборка + случайный поднабор признаков в каждом узле + лучший
порог среди них. ExtraTrees: без bootstrap (или с), и порог в каждом узле
выбирается СЛУЧАЙНО, не лучший. Выше разброс, меньше дисперсия ансамбля.

Концепция 283: Регрессионное дерево.
То же CART, но в листе — среднее (или медиана) y, критерий — MSE/MAE.
Кусочно-постоянная функция.

Концепция 284: Oblique splits (косые расщепления, концепт).
Вместо "x_j <= t" разрешаем "w^T x <= t" — гиперплоскость, не ось.
Сильно мощнее на коррелированных признаках, но дороже обучать (нужен
оптимизатор). Примеры: HHCART, OC1, Forest-RC.

Концепция 285: Поверхностный взгляд на дерево.
get_depth и get_n_leaves у уже обученного DecisionTree показывают
сложность модели; tree_.feature/threshold — массивы решений в узлах.
"""
import numpy as np
from sklearn.datasets import make_moons
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

X, y = make_moons(n_samples=600, noise=0.3, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

# 282: ExtraTrees vs RandomForest
rf = RandomForestClassifier(n_estimators=50, random_state=0).fit(X_tr, y_tr)
et = ExtraTreesClassifier(n_estimators=50, random_state=0).fit(X_tr, y_tr)
print(f"282 RF acc={rf.score(X_te, y_te):.3f}, ExtraTrees acc={et.score(X_te, y_te):.3f}")

# 283: regression tree
rng = np.random.default_rng(0)
xr = np.linspace(0, 10, 200).reshape(-1, 1)
yr = np.sin(xr.ravel()) + rng.normal(0, 0.2, 200)
dtr = DecisionTreeRegressor(max_depth=5, random_state=0).fit(xr, yr)
print(f"283 regression tree R^2={dtr.score(xr, yr):.3f}, n_leaves={dtr.get_n_leaves()}")

# 284: oblique split (концепт) — берём LogisticRegression как 'oblique splitter' на корне
lr = LogisticRegression().fit(X_tr, y_tr)
left_mask = lr.predict(X_tr) == 0
print(f"284 oblique split via LR на корне: левый дочерний размер={int(left_mask.sum())}, правый={int((~left_mask).sum())}")
print(f"     полные oblique-trees (HHCART/OC1) — это рекурсивно тот же приём в каждом узле")

# 285: поверхностный взгляд
print(f"285 RF first tree depth={rf.estimators_[0].get_depth()}, n_leaves={rf.estimators_[0].get_n_leaves()}")
