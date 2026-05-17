"""Концепция 102: Bias-Variance tradeoff.

Bias  - 'недосложность' модели (плохо ловит даже на трейне).
Variance - 'переподгонка' (сильно зависит от выборки).
Идеал: и то, и другое мало. Обычно есть компромисс.
"""
import numpy as np
from sklearn.tree import DecisionTreeRegressor

rng = np.random.default_rng(0)
true_fn = lambda x: np.sin(2 * x)
preds = {1: [], 5: [], 20: []}

x_test = np.linspace(-3, 3, 100).reshape(-1, 1)
y_true = true_fn(x_test.ravel())

for trial in range(50):
    X = rng.uniform(-3, 3, size=(40, 1))
    y = true_fn(X.ravel()) + rng.normal(scale=0.3, size=40)
    for depth in preds:
        m = DecisionTreeRegressor(max_depth=depth, random_state=trial).fit(X, y)
        preds[depth].append(m.predict(x_test))

for depth, ps in preds.items():
    ps = np.array(ps)
    bias2 = ((ps.mean(0) - y_true) ** 2).mean()
    var = ps.var(0).mean()
    print(f"depth={depth:>2}: bias^2={bias2:.3f}  variance={var:.3f}")
