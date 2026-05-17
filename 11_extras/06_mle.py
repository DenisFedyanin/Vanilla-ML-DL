"""Концепция 113: MLE - метод максимального правдоподобия.

Подбираем параметры так, чтобы наблюдаемые данные были максимально вероятны.
Для нормального распределения MLE даёт mu = mean(X), sigma^2 = var(X).
"""
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(loc=5, scale=2, size=1000)

mu_mle = X.mean()
sigma2_mle = X.var()
print(f"Истинные: mu=5, sigma^2=4")
print(f"MLE   :  mu={mu_mle:.3f}, sigma^2={sigma2_mle:.3f}")
