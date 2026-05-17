"""Концепция 12: Центральная предельная теорема (ЦПТ).

Сумма (или среднее) большого числа независимых случайных величин
имеет распределение, близкое к нормальному, независимо от исходного.
"""
import numpy as np

rng = np.random.default_rng(0)
N_TRIALS = 5000

# Берем средние от выборок из РАВНОМЕРНОГО распределения
for sample_size in [1, 2, 5, 30]:
    means = rng.uniform(0, 1, size=(N_TRIALS, sample_size)).mean(axis=1)
    print(f"размер выборки {sample_size:>2}: mean={means.mean():.3f} std={means.std():.3f}")

print("При росте размера выборки std среднего падает как 1/sqrt(N)")
