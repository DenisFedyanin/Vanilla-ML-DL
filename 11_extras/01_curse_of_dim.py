"""Концепция 108: Проклятие размерности.

В высокой размерности 'расстояния' теряют смысл - все точки оказываются
примерно равноудалёнными.
"""
import numpy as np

rng = np.random.default_rng(0)
for D in [2, 5, 10, 100, 1000]:
    X = rng.uniform(size=(500, D))
    d = np.linalg.norm(X[:, None] - X[None], axis=2)
    d = d[d > 0]
    print(f"D={D:>4}: min/max расстояния = {d.min():.2f}/{d.max():.2f}  "
          f"std/mean = {d.std()/d.mean():.3f}")
print("В большой размерности отношение std/mean -> 0: расстояния 'схлопываются'.")
