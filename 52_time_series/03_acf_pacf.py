"""Раздел 52 — Time series. Файл 3: ACF и PACF.

Концепция: Автокорреляция (ACF).
ACF(k) = corr(Y_t, Y_{t-k}). Показывает, как значение в момент t связано
со значением k шагов назад. Для AR(1) с alpha>0 ACF убывает геометрически.

Концепция: Частичная автокорреляция (PACF).
PACF(k) = корреляция Y_t и Y_{t-k} после удаления влияния промежуточных
лагов Y_{t-1}, ..., Y_{t-k+1}. Реализация: регрессируем Y_t на 1..k лагов,
PACF(k) = последний коэффициент.

Концепция: Использование ACF/PACF.
AR(p): PACF обрывается на лаге p, ACF убывает геометрически.
MA(q): ACF обрывается на q, PACF убывает геометрически.
Это даёт правило выбора порядков ARMA по виду графиков.

Концепция: Доверительный интервал.
Для белого шума ACF(k) ~ N(0, 1/T). Интервал ±2/sqrt(T). Значения вне —
значимая корреляция.

Концепция: AR(1) ряд для демо.
Y_t = alpha*Y_{t-1} + e. Теоретически ACF(k) = alpha^k, PACF(1) = alpha,
PACF(k)=0 для k>1.
"""
import numpy as np

rng = np.random.default_rng(0)
T = 1000
alpha = 0.7

y = np.zeros(T)
for i in range(1, T):
    y[i] = alpha * y[i - 1] + rng.normal()


def acf(x, max_lag=10):
    x = x - x.mean()
    denom = (x * x).sum()
    return np.array([(x[:T - k] * x[k:]).sum() / denom for k in range(max_lag + 1)])


def pacf(x, max_lag=10):
    out = [1.0]
    for k in range(1, max_lag + 1):
        Y = x[k:]
        X = np.column_stack([np.ones_like(Y)] + [x[k - j:T - j] for j in range(1, k + 1)])
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        out.append(beta[-1])
    return np.array(out)


a = acf(y, max_lag=10)
p = pacf(y, max_lag=10)
ci = 2 / np.sqrt(T)
print(f"AR(1) alpha={alpha}, T={T}")
print(f"ACF  (теор alpha^k):     {a.round(3)}")
print(f"     (теор)              {np.array([alpha ** k for k in range(11)]).round(3)}")
print(f"PACF (только лаг 1 ≠ 0): {p.round(3)}")
print(f"Доверительный интервал ±{ci:.3f}")
print(f"Значимые PACF лаги: {[i for i, v in enumerate(p) if abs(v) > ci and i > 0]}")
