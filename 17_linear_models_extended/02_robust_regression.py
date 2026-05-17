"""Раздел 17 — Линейные модели (расширенно).

Файл 2: робастные регрессии (устойчивы к выбросам).

Концепция 250: Huber.
Квадратичная функция потерь у нуля, линейная далеко. Не "уплывает" из-за
одного гигантского остатка. Параметр epsilon переключает режимы.

Концепция 251: RANSAC.
Случайно выбираем минимальную подвыборку, фитим, оцениваем число inliers.
Повторяем много раз, выбираем модель с наибольшим числом inliers.

Концепция 252: Theil-Sen.
Берём медиану наклонов по всем парам точек. Очень устойчив (до ~29.3%
выбросов).

Концепция 253: Quantile regression.
Минимизирует не среднюю, а q-квантильную ошибку (pinball loss). Можно
строить условные интервалы (10%-90%) вместо одного среднего.
"""
import numpy as np
from sklearn.linear_model import HuberRegressor, LinearRegression, QuantileRegressor, RANSACRegressor, TheilSenRegressor

rng = np.random.default_rng(0)
n = 100
X = np.linspace(0, 10, n).reshape(-1, 1)
y = 2 * X.ravel() + 1 + rng.normal(0, 0.5, n)
# добавим выбросы
y[::15] += 20

ols = LinearRegression().fit(X, y)
print(f"     OLS coef={ols.coef_[0]:.3f}, intercept={ols.intercept_:.3f} (теор: 2 и 1)")

# 250: Huber
hub = HuberRegressor().fit(X, y)
print(f"250 Huber coef={hub.coef_[0]:.3f}, intercept={hub.intercept_:.3f}")

# 251: RANSAC
ran = RANSACRegressor(random_state=0).fit(X, y)
print(f"251 RANSAC coef={ran.estimator_.coef_[0]:.3f}, intercept={ran.estimator_.intercept_:.3f}")

# 252: Theil-Sen
ts = TheilSenRegressor(random_state=0).fit(X, y)
print(f"252 Theil-Sen coef={ts.coef_[0]:.3f}, intercept={ts.intercept_:.3f}")

# 253: Quantile regression — медиана (q=0.5)
qr_med = QuantileRegressor(quantile=0.5, alpha=0, solver="highs").fit(X, y)
qr_lo = QuantileRegressor(quantile=0.1, alpha=0, solver="highs").fit(X, y)
qr_hi = QuantileRegressor(quantile=0.9, alpha=0, solver="highs").fit(X, y)
print(f"253 Quantile q=0.5 coef={qr_med.coef_[0]:.3f}, q=0.1 coef={qr_lo.coef_[0]:.3f}, q=0.9 coef={qr_hi.coef_[0]:.3f}")
