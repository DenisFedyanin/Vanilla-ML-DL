"""Раздел 52 — Time series. Файл 1: Декомпозиция временного ряда.

Концепция: Компоненты ряда.
Y_t = T_t + S_t + R_t (аддитивная) или Y_t = T_t * S_t * R_t (мультипликативная).
T — тренд, S — сезонность, R — остаток. Понимание структуры помогает с
прогнозом и аномалиями.

Концепция: Тренд через moving average.
T_t = среднее по окну [t-w/2, t+w/2]. Сглаживает быстрые колебания. Чем больше
окно, тем глаже тренд. На концах ряда — NaN или укороченное окно.

Концепция: Сезонность.
После вычитания тренда D_t = Y_t - T_t. Усредняем D_t по позиции в сезоне
(понедельники с понедельниками, январи с январями) — получаем S_t.

Концепция: Остаток.
R_t = Y_t - T_t - S_t. В идеале — белый шум: нулевое среднее, постоянная
дисперсия, отсутствие автокорреляции. Если в остатке есть структура —
декомпозиция упустила её.

Концепция: STL.
Loess-based STL (Seasonal-Trend-Loess) — более устойчивый аналог: робастные
локальные регрессии вместо скользящего среднего. Хорошо работает при выбросах.
"""
import numpy as np

rng = np.random.default_rng(0)
T = 120  # 120 точек
period = 12

t = np.arange(T)
trend_true = 0.05 * t
season_true = 2.0 * np.sin(2 * np.pi * t / period)
noise = rng.normal(scale=0.5, size=T)
y = trend_true + season_true + noise

# Тренд: центрированное скользящее среднее окна = period
w = period
trend_est = np.full(T, np.nan)
half = w // 2
for i in range(half, T - half):
    trend_est[i] = y[i - half:i + half + 1].mean()

# Заполним края константой
trend_est[:half] = trend_est[half]
trend_est[-half:] = trend_est[-half - 1]

# Сезонность: средняя по позиции в сезоне
detrended = y - trend_est
season_est = np.zeros(T)
for p in range(period):
    mask = (t % period) == p
    season_est[mask] = detrended[mask].mean()
season_est -= season_est.mean()             # центрируем

# Остаток
resid = y - trend_est - season_est

print(f"Размер ряда: {T}, период: {period}")
print(f"Тренд: y[0..5]_true={trend_true[:6].round(2)}  est={trend_est[:6].round(2)}")
print(f"Сезон: первые 12 значений est = {season_est[:12].round(2)}")
print(f"Resid: mean={resid.mean():+.3f}, std={resid.std():.3f}  (≈ N(0, 0.5))")
print(f"corr(season_true, season_est) = {np.corrcoef(season_true, season_est)[0,1]:.3f}")
