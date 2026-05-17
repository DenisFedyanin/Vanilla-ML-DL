"""Раздел 54 — Bayesian ML. Файл 3: Bayesian Linear Regression.

Концепция: Модель.
y = X w + eps, eps ~ N(0, sigma^2 I). Prior на веса: w ~ N(m_0, S_0).
Задача — получить полное posterior p(w | X, y), а не только точечную оценку.

Концепция: Закрытая форма posterior (Gaussian prior + Gaussian likelihood).
S_n^{-1} = S_0^{-1} + (1/sigma^2) X^T X
m_n     = S_n * (S_0^{-1} m_0 + (1/sigma^2) X^T y)
Posterior: w ~ N(m_n, S_n). Conjugate-результат для нормального шума.

Концепция: Связь с Ridge.
При m_0 = 0, S_0 = (1/alpha) I получаем m_n = (X^T X + alpha*sigma^2 I)^{-1} X^T y.
Это в точности ridge-решение с регуляризацией lambda = alpha*sigma^2.

Концепция: Posterior predictive.
Для нового x*: p(y* | x*, D) = N(x*^T m_n, sigma^2 + x*^T S_n x*).
Второе слагаемое — эпистемическая неопределённость (от незнания w),
первое — шум данных (алеаторная).

Концепция: Зачем это нужно.
Получаем интервалы предсказания, а не только точку. Active learning:
выбираем x с максимальной epistemic uncertainty. Bayesian optimization.
"""
import numpy as np

rng = np.random.default_rng(0)

# Истинные параметры
w_true = np.array([1.5, -2.0, 0.5])
sigma = 0.5
n = 40
X = rng.normal(size=(n, 3))
y = X @ w_true + rng.normal(scale=sigma, size=n)

# Prior: m_0 = 0, S_0 = (1/alpha) I
alpha = 0.5
S0_inv = alpha * np.eye(3)
m0 = np.zeros(3)

# Posterior
Sn_inv = S0_inv + (X.T @ X) / sigma ** 2
Sn = np.linalg.inv(Sn_inv)
mn = Sn @ (S0_inv @ m0 + (X.T @ y) / sigma ** 2)

print(f"Истинные w     : {w_true}")
print(f"Posterior mean : {mn.round(3)}")
print(f"Posterior std  : {np.sqrt(np.diag(Sn)).round(3)}")

# Ridge для сравнения
lam = alpha * sigma ** 2
w_ridge = np.linalg.solve(X.T @ X + lam * np.eye(3), X.T @ y)
print(f"Ridge (lam={lam}): {w_ridge.round(3)}  (совпадает с posterior mean)")

# Posterior predictive для 3 новых точек
X_new = rng.normal(size=(3, 3))
mean_pred = X_new @ mn
var_pred = sigma ** 2 + np.einsum("ij,jk,ik->i", X_new, Sn, X_new)
y_true_new = X_new @ w_true
print("Posterior predictive (mean ± std)  vs  истинное y:")
for i in range(3):
    print(f"  {mean_pred[i]:+.3f} ± {np.sqrt(var_pred[i]):.3f}   true={y_true_new[i]:+.3f}")
