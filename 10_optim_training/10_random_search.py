"""Концепция 107: Random Search.

Вместо полной сетки сэмплируем случайные комбинации. Часто эффективнее grid'a,
особенно когда важных параметров мало - быстро находим хорошее значение.
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=300, noise=0.3, random_state=0)
dist = {"max_depth": list(range(2, 20)),
        "min_samples_split": list(range(2, 20))}
rs = RandomizedSearchCV(DecisionTreeClassifier(random_state=0),
                        dist, n_iter=20, cv=3, random_state=0).fit(X, y)
print("Лучшие параметры:", rs.best_params_)
print(f"Лучший CV score : {rs.best_score_:.3f}")
