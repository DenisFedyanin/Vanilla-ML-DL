"""Концепция 106: Grid Search.

Перебираем все комбинации гиперпараметров на сетке, выбираем лучшую по CV.
Просто и надёжно, но дорого при большом числе параметров.
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=300, noise=0.3, random_state=0)
grid = {"max_depth": [2, 3, 5, 8], "min_samples_split": [2, 5, 10]}
gs = GridSearchCV(DecisionTreeClassifier(random_state=0), grid, cv=3).fit(X, y)
print("Лучшие параметры:", gs.best_params_)
print(f"Лучший CV score : {gs.best_score_:.3f}")
