"""Раздел 52 — Time series. Файл 2: Стационарность.

Концепция: Стационарность (строгая и слабая).
Строгая: совместное распределение (Y_t1..Y_tk) не меняется при сдвиге времени.
Слабая (нам нужна): E[Y_t] и Var(Y_t) — константы, Cov(Y_t, Y_{t+k}) зависит
только от k. Большинство методов TS требуют стационарности.

Концепция: Дифференцирование.
delta Y_t = Y_t - Y_{t-1} убирает тренд (часто превращает random walk в шум).
Сезонное дифференцирование: Y_t - Y_{t-period} убирает сезон.
ARIMA(p,d,q) использует d таких разностей.

Концепция: Лог-преобразование.
log Y_t стабилизирует растущую дисперсию (мультипликативный шум -> аддитивный).
Применимо только для строго положительных рядов.

Концепция: ADF (Augmented Dickey-Fuller).
H0: ряд имеет единичный корень (нестационарен). Регрессия:
delta Y_t = a + b*t + rho*Y_{t-1} + sum gamma_i*delta Y_{t-i} + e.
Тест-статистика: rho_hat / SE(rho_hat). Чем отрицательнее — тем сильнее отвергаем H0.

Концепция: KPSS.
Дополняющий тест: H0 — ряд стационарен. Полезно использовать ADF и KPSS
вместе для уверенности.
"""
import numpy as np

rng = np.random.default_rng(0)
T = 300

# Random walk (нестационарен)
rw = np.cumsum(rng.normal(size=T))
# Стационарный AR(0.5)
y = np.zeros(T)
for i in range(1, T):
    y[i] = 0.5 * y[i - 1] + rng.normal(scale=1)

print(f"Random walk: mean[:100]={rw[:100].mean():.2f}, mean[200:]={rw[200:].mean():.2f}")
print(f"AR(0.5)    : mean[:100]={y[:100].mean():.2f}, mean[200:]={y[200:].mean():.2f}")


def adf_stat(x):
    """Упрощённый ADF: регрессируем delta Y на Y_{t-1}, без лагов и без тренда."""
    dy = np.diff(x)
    y_lag = x[:-1]
    A = np.column_stack([np.ones_like(y_lag), y_lag])
    beta, *_ = np.linalg.lstsq(A, dy, rcond=None)
    resid = dy - A @ beta
    sigma2 = (resid ** 2).sum() / (len(dy) - 2)
    var_b = sigma2 * np.linalg.inv(A.T @ A)[1, 1]
    return beta[1] / np.sqrt(var_b)


print(f"ADF(random walk) = {adf_stat(rw):+.2f}  (близко к 0 — единичный корень)")
print(f"ADF(AR(0.5))     = {adf_stat(y):+.2f}  (сильно отрицательное — стационарно)")

# Дифференцирование делает RW стационарным
rw_diff = np.diff(rw)
print(f"diff(rw): mean={rw_diff.mean():+.3f}, std={rw_diff.std():.3f}  (≈ белый шум)")

# Лог-трансформация для растущей дисперсии
exp_series = np.exp(0.005 * np.arange(T) + rng.normal(scale=0.3, size=T))
log_series = np.log(exp_series)
print(f"std до log: {exp_series.std():.2f},  после log: {log_series.std():.2f}")
