"""Концепция 110: Permutation Importance.

Перемешиваем значения одной фичи и смотрим, насколько упал скор.
Чем сильнее упал - тем важнее была фича. Работает с любой моделью.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
rf = RandomForestClassifier(random_state=0).fit(X, y)
base = rf.score(X, y)
rng = np.random.default_rng(0)
names = ["sepal_len", "sepal_wid", "petal_len", "petal_wid"]
for i, n in enumerate(names):
    Xp = X.copy()
    rng.shuffle(Xp[:, i])
    drop = base - rf.score(Xp, y)
    print(f"{n:>12}: важность = {drop:.3f}")
