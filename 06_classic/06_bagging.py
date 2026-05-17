"""Концепция 54: Bagging (Bootstrap Aggregating).

Берём N бутстрэп-выборок (с возвратом), обучаем по дереву на каждой,
усредняем предсказания. Уменьшает дисперсию модели.
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=300, noise=0.3, random_state=0)
rng = np.random.default_rng(0)

trees = []
for i in range(20):
    idx = rng.choice(len(X), len(X), replace=True)  # bootstrap
    t = DecisionTreeClassifier(max_depth=5).fit(X[idx], y[idx])
    trees.append(t)

preds = np.mean([t.predict(X) for t in trees], axis=0) > 0.5
acc = (preds == y).mean()
print(f"Bagging accuracy: {acc:.3f}")
