"""Раздел 17 — Линейные модели (расширенно).

Файл 1: разные solver'ы для регуляризованной регрессии.

Концепция 247: Ridge — closed form vs SGD.
Замкнутая формула: w = (X^T X + alpha*I)^-1 X^T y. Точно, но O(d^3).
SGDRegressor с penalty='l2' приближённо ищет тот же оптимум градиентным спуском.

Концепция 248: Lasso через coordinate descent.
Минимизирует ||y - Xw||^2 + alpha*||w||_1. Из-за L1 решение разрежено
(многие w_i = 0 -> отбор признаков). Sklearn использует координатный спуск.

Концепция 249: ElasticNetCV.
Lasso + Ridge: alpha*(l1_ratio*||w||_1 + (1-l1_ratio)/2*||w||_2^2). CV-версия
автоматически подбирает alpha и l1_ratio.
"""
import numpy as np
from sklearn.linear_model import ElasticNetCV, Lasso, SGDRegressor

rng = np.random.default_rng(0)
n, d = 200, 10
X = rng.normal(size=(n, d))
w_true = np.array([3.0, -2.0, 1.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
y = X @ w_true + rng.normal(0, 0.5, size=n)

# 247: Ridge closed-form vs SGD
alpha = 1.0
w_cf = np.linalg.solve(X.T @ X + alpha * np.eye(d), X.T @ y)
sgd = SGDRegressor(penalty="l2", alpha=alpha / n, learning_rate="invscaling", max_iter=2000, random_state=0)
sgd.fit(X, y)
w_sgd = sgd.coef_
print(f"247 Ridge closed-form w[:4]={w_cf[:4].round(3)}")
print(f"     Ridge SGD          w[:4]={w_sgd[:4].round(3)}")

# 248: Lasso coordinate descent (sklearn)
lasso = Lasso(alpha=0.1, max_iter=10_000).fit(X, y)
n_zero = int((lasso.coef_ == 0).sum())
print(f"248 Lasso coefs={lasso.coef_.round(3)}, n_zero={n_zero}/{d}")

# 249: ElasticNetCV
encv = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5, random_state=0).fit(X, y)
print(f"249 ElasticNetCV best_alpha={encv.alpha_:.3f}, l1_ratio={encv.l1_ratio_}, coefs[:4]={encv.coef_[:4].round(3)}")
