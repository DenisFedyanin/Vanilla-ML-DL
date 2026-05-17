"""Концепция 62: Voting classifier.

Объединяем несколько разных моделей. Hard voting - голосование большинством,
soft voting - усреднение вероятностей. Часто работает лучше отдельных моделей.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=300, noise=0.3, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0)

models = [("lr", LogisticRegression()),
          ("tree", DecisionTreeClassifier(max_depth=5)),
          ("knn", KNeighborsClassifier())]
for name, m in models:
    m.fit(Xtr, ytr)
    print(f"{name}: {m.score(Xte, yte):.3f}")

vc = VotingClassifier(models, voting="soft").fit(Xtr, ytr)
print(f"voting (soft): {vc.score(Xte, yte):.3f}")
