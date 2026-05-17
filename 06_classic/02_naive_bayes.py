"""Концепция 50: Gaussian Naive Bayes.

Предполагаем независимость признаков. Для каждого класса оцениваем
mean и var. Предсказание - класс с максимальной апостериорной вероятностью.
"""
import numpy as np

rng = np.random.default_rng(0)
X0 = rng.normal(loc=0, scale=1, size=(100, 2))
X1 = rng.normal(loc=3, scale=1, size=(100, 2))
X = np.vstack([X0, X1])
y = np.array([0] * 100 + [1] * 100)


def fit(X, y):
    params = {}
    for c in np.unique(y):
        Xc = X[y == c]
        params[c] = (Xc.mean(0), Xc.var(0) + 1e-9, (y == c).mean())
    return params


def predict(x, params):
    best, best_log = None, -np.inf
    for c, (mu, var, pi) in params.items():
        log_p = np.log(pi) - 0.5 * np.sum(np.log(2 * np.pi * var)) \
                            - 0.5 * np.sum((x - mu) ** 2 / var)
        if log_p > best_log:
            best, best_log = c, log_p
    return best


params = fit(X, y)
acc = np.mean([predict(x, params) for x in X] == y)
print(f"Train accuracy: {acc:.3f}")
