"""Концепция 56: AdaBoost - очень упрощённая реализация (decision stumps).

Веса примеров увеличиваются для тех, что мы неправильно предсказали,
и следующая модель фокусируется на трудных примерах.
"""
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 2))
y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)

w = np.ones(len(X)) / len(X)
stumps = []
for t in range(10):
    best = (None, None, None, np.inf)
    for f in range(2):
        for thr in np.linspace(-2, 2, 20):
            for s in [1, -1]:
                pred = np.where(s * (X[:, f] - thr) > 0, 1, -1)
                err = np.sum(w * (pred != y))
                if err < best[3]:
                    best = (f, thr, s, err)
    f, thr, s, err = best
    err = max(err, 1e-9)
    alpha = 0.5 * np.log((1 - err) / err)
    pred = np.where(s * (X[:, f] - thr) > 0, 1, -1)
    w *= np.exp(-alpha * y * pred); w /= w.sum()
    stumps.append((f, thr, s, alpha))

agg = np.zeros(len(X))
for f, thr, s, a in stumps:
    agg += a * np.where(s * (X[:, f] - thr) > 0, 1, -1)
acc = ((np.sign(agg) == y)).mean()
print(f"AdaBoost (10 пней) accuracy: {acc:.3f}")
