"""Концепция 51: Decision Tree (с нуля, для понимания).

Рекурсивно ищем лучший split по Gini и строим дерево до заданной глубины.
"""
import numpy as np


def gini(y):
    if len(y) == 0:
        return 0
    _, c = np.unique(y, return_counts=True)
    return 1 - np.sum((c / c.sum()) ** 2)


def best_split(X, y):
    best = (None, None, np.inf)
    for f in range(X.shape[1]):
        for thr in np.unique(X[:, f]):
            l = y[X[:, f] <= thr]; r = y[X[:, f] > thr]
            if len(l) == 0 or len(r) == 0:
                continue
            g = (len(l) * gini(l) + len(r) * gini(r)) / len(y)
            if g < best[2]:
                best = (f, thr, g)
    return best


def build(X, y, depth=0, max_depth=3):
    if depth == max_depth or len(np.unique(y)) == 1:
        vals, c = np.unique(y, return_counts=True)
        return {"leaf": vals[c.argmax()]}
    f, thr, _ = best_split(X, y)
    if f is None:
        vals, c = np.unique(y, return_counts=True)
        return {"leaf": vals[c.argmax()]}
    mask = X[:, f] <= thr
    return {"f": f, "thr": thr,
            "L": build(X[mask], y[mask], depth + 1, max_depth),
            "R": build(X[~mask], y[~mask], depth + 1, max_depth)}


def predict_one(tree, x):
    while "leaf" not in tree:
        tree = tree["L"] if x[tree["f"]] <= tree["thr"] else tree["R"]
    return tree["leaf"]


rng = np.random.default_rng(0)
X = rng.normal(size=(80, 2))
y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)  # XOR
tree = build(X, y, max_depth=3)
acc = np.mean([predict_one(tree, x) for x in X] == y)
print(f"Accuracy дерева глубины 3 на XOR: {acc:.3f}")
