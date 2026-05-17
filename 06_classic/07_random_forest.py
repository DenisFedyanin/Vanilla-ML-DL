"""Концепция 55: Random Forest.

Bagging + случайный выбор признаков в каждом узле. Один из самых надёжных
'из коробки' алгоритмов.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=400, noise=0.25, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0)
rf = RandomForestClassifier(n_estimators=100, random_state=0).fit(Xtr, ytr)
print(f"Train acc: {rf.score(Xtr, ytr):.3f}")
print(f"Test  acc: {rf.score(Xte, yte):.3f}")
