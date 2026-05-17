"""Раздел 31 — Лоссы (расширенно). Файл 1: регрессионные.

Концепция 342: MSE = mean((y-yhat)^2).
Гладкий, выпуклый. Сильно штрафует большие ошибки -> чувствителен к выбросам.

Концепция 343: MAE = mean(|y-yhat|).
Устойчив к выбросам. Не гладкий в нуле; градиент = sign(error).

Концепция 344: Huber loss.
Гибрид: квадратичный при |error|<=delta, линейный при больших.
Гладкий + устойчивый к выбросам. Стандарт в робастной регрессии.

Концепция 345: Quantile / Pinball loss.
L_q(e) = max(q*e, (q-1)*e). Минимизация даёт q-квантиль (а не среднее).
Применяется в квантильной регрессии, прогнозе интервалов.

Концепция 346: Log-cosh.
log(cosh(error)) ~ 0.5*error^2 при малых и |error| при больших.
Везде гладкая, в т.ч. вторая производная — удобно для оптимизаторов 2-го порядка.
"""
import numpy as np

rng = np.random.default_rng(0)
y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
yhat = np.array([1.2, 1.8, 3.5, 3.6, 9.0])     # одно сильное выпадание (9.0 вместо 5)
err = yhat - y

# === 342: MSE ===
mse = np.mean(err ** 2)
print(f"342 MSE = {mse:.3f} (выброс даёт огромный вклад: {err[-1] ** 2:.1f})")

# === 343: MAE ===
mae = np.mean(np.abs(err))
print(f"343 MAE = {mae:.3f}")

# === 344: Huber loss ===
def huber(e, delta=1.0):
    abs_e = np.abs(e)
    quad = 0.5 * e ** 2
    lin = delta * (abs_e - 0.5 * delta)
    return np.where(abs_e <= delta, quad, lin)

print(f"344 Huber(d=1.0) = {huber(err).mean():.3f}")
print(f"    Huber(d=2.0) = {huber(err, 2.0).mean():.3f}")

# === 345: Quantile / Pinball ===
def pinball(e, q):
    return np.maximum(q * e, (q - 1) * e)

for q in (0.1, 0.5, 0.9):
    print(f"345 Pinball q={q}: loss = {pinball(err, q).mean():.3f}")

# === 346: Log-cosh ===
def logcosh(e):
    # численно-стабильно: log(cosh(e)) = |e| + log(1+exp(-2|e|)) - log 2
    a = np.abs(e)
    return a + np.log1p(np.exp(-2 * a)) - np.log(2)

print(f"346 Log-cosh = {logcosh(err).mean():.3f}")

# Сравним чувствительность к выбросу: посчитаем без последнего значения
for name, fn in [("MSE", lambda e: e ** 2), ("MAE", lambda e: np.abs(e)),
                 ("Huber1", lambda e: huber(e, 1.0)), ("Logcosh", logcosh)]:
    full = fn(err).mean()
    no_out = fn(err[:-1]).mean()
    print(f"   {name}: с выбросом={full:.3f}, без выброса={no_out:.3f}, отношение={full / no_out:.1f}x")
