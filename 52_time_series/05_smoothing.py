"""Раздел 52 — Time series. Файл 5: Экспоненциальное сглаживание.

Концепция: Simple Exponential Smoothing (SES).
L_t = alpha * Y_t + (1 - alpha) * L_{t-1}. Прогноз: y_hat_{t+h} = L_t.
Подходит для рядов без тренда и сезонности. Большие alpha — быстрая реакция,
малые — сильное сглаживание.

Концепция: Holt's linear trend.
Добавляем компонент тренда B_t.
  L_t = alpha * Y_t + (1 - alpha) * (L_{t-1} + B_{t-1})
  B_t = beta  * (L_t - L_{t-1}) + (1 - beta) * B_{t-1}
Прогноз: y_hat_{t+h} = L_t + h * B_t. Подходит для рядов с трендом.

Концепция: Holt-Winters (аддитивная сезонность).
Добавляем сезонный компонент S_t периода m.
  L_t = alpha*(Y_t - S_{t-m}) + (1-alpha)*(L_{t-1} + B_{t-1})
  B_t = beta*(L_t - L_{t-1}) + (1-beta)*B_{t-1}
  S_t = gamma*(Y_t - L_t)    + (1-gamma)*S_{t-m}
Прогноз: y_hat_{t+h} = L_t + h*B_t + S_{t-m+((h-1) mod m)+1}.

Концепция: Мультипликативная версия.
Если сезонная амплитуда растёт с уровнем ряда, использовать умножение вместо
сложения для S и B.

Концепция: ETS notation.
Error-Trend-Seasonal: (A/M, N/A/Ad, N/A/M). SES = (A,N,N), Holt = (A,A,N),
Holt-Winters add = (A,A,A). Унифицирует всё семейство.
"""
import numpy as np

rng = np.random.default_rng(0)
T = 120
period = 12
t = np.arange(T)
y = 10 + 0.1 * t + 3 * np.sin(2 * np.pi * t / period) + rng.normal(scale=0.5, size=T)

# === SES ===
alpha = 0.3
L = np.zeros(T)
L[0] = y[0]
for i in range(1, T):
    L[i] = alpha * y[i] + (1 - alpha) * L[i - 1]
print(f"SES last L = {L[-1]:.2f}  (прогноз = это число)")

# === Holt ===
a, b = 0.3, 0.1
L = np.zeros(T); B = np.zeros(T)
L[0] = y[0]; B[0] = y[1] - y[0]
for i in range(1, T):
    L[i] = a * y[i] + (1 - a) * (L[i - 1] + B[i - 1])
    B[i] = b * (L[i] - L[i - 1]) + (1 - b) * B[i - 1]
print(f"Holt: L={L[-1]:.2f}, B={B[-1]:+.3f}  forecast h=5: {L[-1] + 5 * B[-1]:.2f}")

# === Holt-Winters (additive) ===
a, b, g = 0.3, 0.1, 0.3
L = np.zeros(T); B = np.zeros(T); S = np.zeros(T)
L[0] = np.mean(y[:period])
B[0] = (np.mean(y[period:2 * period]) - np.mean(y[:period])) / period
S[:period] = y[:period] - L[0]
for i in range(period, T):
    L[i] = a * (y[i] - S[i - period]) + (1 - a) * (L[i - 1] + B[i - 1])
    B[i] = b * (L[i] - L[i - 1]) + (1 - b) * B[i - 1]
    S[i] = g * (y[i] - L[i]) + (1 - g) * S[i - period]

h_max = 12
fc = [L[-1] + h * B[-1] + S[-period + ((h - 1) % period)] for h in range(1, h_max + 1)]
print(f"Holt-Winters forecast next year:")
print(f"  {np.round(fc, 2)}")
print(f"Last observed year: {np.round(y[-12:], 2)}")
