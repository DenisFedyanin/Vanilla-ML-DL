"""Концепция 109: Feature importance (через Random Forest).

Считаем, как сильно фича уменьшает gini/entropy при разбиениях.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
rf = RandomForestClassifier(random_state=0).fit(X, y)
names = ["sepal_len", "sepal_wid", "petal_len", "petal_wid"]
for n, imp in sorted(zip(names, rf.feature_importances_), key=lambda x: -x[1]):
    print(f"{n:>12}: {imp:.3f}")
