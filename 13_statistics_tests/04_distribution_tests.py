"""Раздел 13 — Статистические тесты.

Файл 4: тесты согласия и независимости.

Концепция 167: Kolmogorov-Smirnov (одновыборочный).
Сравниваем эмпирическую CDF с теоретической. Статистика D = sup|F_emp(x) - F_теор(x)|.
Не зависит от формы распределения под H0.

Концепция 168: Shapiro-Wilk (концепт).
Тест на нормальность. W = (sum a_i x_(i))^2 / sum (x - mean)^2. Очень мощный
для нормальности, но реализация полного алгоритма выходит за рамки. Просто
покажем W приближённо: корреляция упорядоченной выборки с теоретическими
квантилями нормали (Q-Q идея).

Концепция 169: Хи-квадрат критерий согласия.
Сравниваем наблюдаемые частоты по k категориям с ожидаемыми.
chi2 = sum (O_i - E_i)^2 / E_i ~ chi^2(k - 1 - p), где p — оценённых параметров.

Концепция 170: Хи-квадрат критерий независимости (таблицы сопряжённости).
Для таблицы r x c ожидаемое E_ij = (row_i*col_j)/N. chi2 ~ chi^2((r-1)(c-1)).
"""
import math

import numpy as np

rng = np.random.default_rng(0)

# 167: KS против N(0,1)
x = rng.normal(0, 1, size=200)
x_sorted = np.sort(x)
n = len(x)
emp = np.arange(1, n + 1) / n


def norm_cdf(z):
    return 0.5 * (1.0 + np.array([math.erf(zi / np.sqrt(2)) for zi in z]))


theor = norm_cdf(x_sorted)
D = np.max(np.abs(emp - theor))
print(f"167 KS test N(0,1): D={D:.4f} (порог 1.36/sqrt(n)≈{1.36/np.sqrt(n):.3f} при alpha=0.05)")

# Проверим против явно ненормального (экспонента)
y = rng.exponential(scale=1.0, size=200)
y_sorted = np.sort(y)
emp_y = np.arange(1, n + 1) / n
theor_y = norm_cdf(y_sorted)
D_y = np.max(np.abs(emp_y - theor_y))
print(f"     vs Exp(1) ~ должно быть много больше: D={D_y:.4f}")

# 168: Shapiro-Wilk упрощённо: корреляция отсортированной выборки с теоретическими z-квантилями
# теоретические квантили: норм.обратная функция приближённо через Newton/Erfinv? используем np сэмплинг N(0,1) сортированных
def approx_normal_quantiles(n):
    # квантили (i-0.5)/n через inverse через bisection с norm_cdf
    p = (np.arange(1, n + 1) - 0.5) / n
    # invert norm_cdf via vectorized: используем np ppf-эквивалент через np.percentile большого N(0,1)
    big = np.sort(rng.standard_normal(size=200_000))
    idx = (p * len(big)).astype(int).clip(0, len(big) - 1)
    return big[idx]


x_n = np.sort(rng.normal(size=100))
q = approx_normal_quantiles(len(x_n))
corr_n = np.corrcoef(x_n, q)[0, 1]
x_e = np.sort(rng.exponential(size=100))
corr_e = np.corrcoef(x_e, q)[0, 1]
print(f"168 Shapiro-like (corr с N-квантилями): нормальное={corr_n:.4f}, экспонента={corr_e:.4f}")

# 169: chi2 GoF — кубик честный?
faces = rng.integers(1, 7, size=600)
O = np.bincount(faces, minlength=7)[1:]
E = np.full(6, 100.0)  # ожидаем по 100
chi2_gof = ((O - E) ** 2 / E).sum()
print(f"169 chi2 GoF (кубик): O={O.tolist()}, chi2={chi2_gof:.3f}, df=5")

# 170: chi2 независимости — пол и предпочтение
table = np.array([[30, 20, 50],   # мужчины
                  [40, 25, 35]])  # женщины
row_tot = table.sum(axis=1, keepdims=True)
col_tot = table.sum(axis=0, keepdims=True)
N = table.sum()
E = row_tot @ col_tot / N
chi2_ind = ((table - E) ** 2 / E).sum()
df = (table.shape[0] - 1) * (table.shape[1] - 1)
print(f"170 chi2 indep: chi2={chi2_ind:.3f}, df={df}")
