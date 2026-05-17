"""Концепция 61: One-vs-Rest для мультикласса.

Имея бинарный классификатор, обучаем K штук: 'класс k против всех остальных'.
Предсказание - класс с наибольшим скором.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
classes = np.unique(y)
models = {}
for c in classes:
    models[c] = LogisticRegression(max_iter=1000).fit(X, (y == c).astype(int))

scores = np.column_stack([models[c].predict_proba(X)[:, 1] for c in classes])
pred = classes[scores.argmax(axis=1)]
print(f"Accuracy OvR на Iris: {(pred == y).mean():.3f}")
