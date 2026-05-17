"""Концепция 63: Stacking (концепт).

Шаг 1: обучаем несколько 'базовых' моделей.
Шаг 2: их предсказания становятся новыми признаками для 'мета-модели'.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=400, noise=0.3, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0)

base = [DecisionTreeClassifier(max_depth=5, random_state=0),
        KNeighborsClassifier(),
        LogisticRegression()]

# для простоты обучаем базовые модели на трейне и собираем мета-фичи
for m in base:
    m.fit(Xtr, ytr)

meta_X = np.column_stack([m.predict_proba(Xtr)[:, 1] for m in base])
meta_X_te = np.column_stack([m.predict_proba(Xte)[:, 1] for m in base])
meta = LogisticRegression().fit(meta_X, ytr)
print(f"Stacking test acc: {meta.score(meta_X_te, yte):.3f}")
