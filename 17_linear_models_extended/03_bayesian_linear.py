"""Раздел 17 — Линейные модели (расширенно).

Файл 3: байесовские линейные регрессии.

Концепция 254: BayesianRidge.
Полностью байесовский ridge: априоры на w (Normal с дисперсией 1/alpha)
и на шум (Gamma на precision). Возвращает posterior среднее коэффициентов
и оценки гиперпараметров alpha (precision весов) и lambda (precision шума).

Концепция 255: ARD (Automatic Relevance Determination).
Каждому весу — свой alpha_i. У "неинформативных" признаков alpha_i становится
огромным, w_i -> 0. Похоже на L1, но через байесовскую разреженность.
"""
import numpy as np
from sklearn.linear_model import ARDRegression, BayesianRidge

rng = np.random.default_rng(0)
n, d = 200, 8
X = rng.normal(size=(n, d))
w_true = np.array([3.0, -2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
y = X @ w_true + rng.normal(0, 0.3, n)

# 254: BayesianRidge
br = BayesianRidge().fit(X, y)
print(f"254 BayesianRidge coefs={br.coef_.round(3)}")
print(f"     intercept={br.intercept_:.3f}, alpha (noise prec)={br.alpha_:.3f}, lambda (weight prec)={br.lambda_:.3f}")

# 255: ARD — должны "обнулиться" неинформативные
ard = ARDRegression().fit(X, y)
print(f"255 ARD coefs={ard.coef_.round(3)}")
print(f"     per-weight precision (1/var_i) первые 4 = {ard.lambda_[:4].round(2)} (большое = неинформативный)")
