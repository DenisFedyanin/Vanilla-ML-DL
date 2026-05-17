"""Концепция 105: Validation curves.

График ошибки в зависимости от ГИПЕРПАРАМЕТРА (например, глубины дерева).
Помогает найти оптимальное значение.
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=400, noise=0.3, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0)
for d in [1, 2, 3, 5, 8, 15]:
    m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ytr)
    print(f"depth={d:>2}: train acc={m.score(Xtr, ytr):.3f}  test acc={m.score(Xte, yte):.3f}")
