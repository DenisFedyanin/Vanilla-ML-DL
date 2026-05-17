"""Раздел 52 — Time series. Файл 4: ARIMA-семья.

Концепция: AR(p) — авторегрессия.
Y_t = c + phi_1*Y_{t-1} + ... + phi_p*Y_{t-p} + e_t. Текущее значение —
линейная комбинация предыдущих p значений плюс шум. Обучается по МНК:
строим матрицу лагов, решаем lstsq.

Концепция: MA(q) — скользящее среднее.
Y_t = mu + e_t + theta_1*e_{t-1} + ... + theta_q*e_{t-q}. Зависимость не от
прошлых Y, а от прошлых ошибок. Обучается итеративно (MLE), не МНК.

Концепция: ARMA(p, q).
Совмещает AR и MA: Y_t = AR-часть + MA-часть. Гибкая модель для стационарных
рядов.

Концепция: ARIMA(p, d, q).
Если ряд нестационарен, применяем дифференцирование d раз -> делаем стационарным,
затем подгоняем ARMA(p,q). 'I' = integrated.

Концепция: SARIMA(p,d,q)(P,D,Q,s).
Добавляет сезонную часть: лаги и разности на расстоянии sезонного периода s.
Например SARIMA(1,0,0)(1,1,0,12) для ежемесячных данных с годовой сезонностью.
"""
import numpy as np

rng = np.random.default_rng(0)
T = 500
phi_true = np.array([0.6, -0.2])
y = np.zeros(T)
for i in range(2, T):
    y[i] = phi_true[0] * y[i - 1] + phi_true[1] * y[i - 2] + rng.normal()

# === AR(p) через lstsq ===
def fit_ar(x, p):
    n = len(x)
    Y = x[p:]
    cols = [np.ones_like(Y)]
    for j in range(p):
        cols.append(x[p - j - 1:n - j - 1])
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return beta


beta = fit_ar(y, 2)
print(f"AR(2) истинные phi = {phi_true}")
print(f"AR(2) оценённые   = {beta[1:].round(3)}, intercept = {beta[0]:+.3f}")

# Прогноз на 5 шагов
y_pred = list(y[-2:])
for _ in range(5):
    y_pred.append(beta[0] + beta[1] * y_pred[-1] + beta[2] * y_pred[-2])
print(f"5-step forecast: {np.array(y_pred[-5:]).round(3)}")

# === ARIMA(1,1,0): берём random walk + AR компонент ===
rw = np.cumsum(rng.normal(size=T))
d_rw = np.diff(rw)
beta_d = fit_ar(d_rw, 1)
print(f"ARIMA(1,1,0): phi на diff(rw) = {beta_d[1]:+.3f}  (должно быть ≈0 — это просто RW)")

# === SARIMA структура: моделируем ряд с годовой сезонностью ===
period = 12
y_s = np.zeros(T)
for i in range(period, T):
    y_s[i] = 0.4 * y_s[i - 1] + 0.5 * y_s[i - period] + rng.normal()

# Sезонное дифференцирование + AR(1)
y_sd = y_s[period:] - y_s[:-period]
beta_s = fit_ar(y_sd, 1)
print(f"SARIMA(1,0,0)(0,1,0,12): AR phi на сезонно-разностном = {beta_s[1]:+.3f}")
