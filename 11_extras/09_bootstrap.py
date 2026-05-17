"""Концепция 116: Bootstrap и доверительный интервал.

Сэмплируем с возвратом из выборки много раз, для каждого ресэмпла считаем
статистику. Распределение этих оценок даёт доверительный интервал.
"""
import numpy as np

rng = np.random.default_rng(0)
data = rng.normal(loc=10, scale=2, size=100)

means = []
for _ in range(2000):
    sample = rng.choice(data, size=len(data), replace=True)
    means.append(sample.mean())
means = np.array(means)

lo, hi = np.percentile(means, [2.5, 97.5])
print(f"Среднее по выборке: {data.mean():.3f}")
print(f"95% доверительный интервал bootstrap: [{lo:.3f}, {hi:.3f}]")
