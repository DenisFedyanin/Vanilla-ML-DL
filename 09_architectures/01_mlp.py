"""Концепция 88: MLP - многослойный перцептрон.

Несколько полносвязных слоёв подряд + нелинейные активации.
Универсальный аппроксиматор: может выучить любую разумную функцию.
"""
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=400, noise=0.25, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0)
mlp = MLPClassifier(hidden_layer_sizes=(16, 16), max_iter=2000,
                    random_state=0).fit(Xtr, ytr)
print(f"Train: {mlp.score(Xtr, ytr):.3f}, Test: {mlp.score(Xte, yte):.3f}")
