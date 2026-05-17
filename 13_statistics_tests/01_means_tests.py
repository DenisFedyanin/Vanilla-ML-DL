"""Раздел 13 — Статистические тесты.

Файл 1: тесты на сравнение СРЕДНИХ.

Концепция 153: Одновыборочный t-тест.
Проверяем H0: mu = mu0 при неизвестной дисперсии. Статистика
t = (mean - mu0) / (s / sqrt(n)) ~ t_{n-1} под H0.

Концепция 154: Двухвыборочный t-тест (равные дисперсии).
Сравниваем две выборки. Считаем pooled sd и t = (m1-m2)/(sp*sqrt(1/n1+1/n2)).
Степени свободы n1+n2-2.

Концепция 155: t-тест Уэлча (неравные дисперсии).
Не пулим: t = (m1-m2)/sqrt(s1^2/n1 + s2^2/n2). Степени свободы по формуле
Велча-Сатеруэйта. Более робастен, чем классический.

Концепция 156: Парный t-тест.
Если выборки парные (до/после), считаем разности d_i = x_i - y_i и
проводим одновыборочный t-тест против 0. Снижает дисперсию.

Концепция 157: z-тест.
Если дисперсия известна или n очень велико, используем z = (mean - mu0)/(sigma/sqrt(n))
со стандартным нормальным распределением вместо t.
"""
import math

import numpy as np

rng = np.random.default_rng(0)


def norm_cdf(z):
    # CDF стандартной нормали через erf
    return 0.5 * (1.0 + math.erf(z / np.sqrt(2)))


# 153: one-sample t-test, H0: mu=0
x = rng.normal(0.3, 1.0, size=40)
t_stat = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
print(f"153 one-sample t: t={t_stat:.3f}, df={len(x)-1}")

# 154: pooled two-sample
a = rng.normal(0.0, 1.0, size=30)
b = rng.normal(0.5, 1.0, size=30)
n1, n2 = len(a), len(b)
sp2 = ((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2)
t2 = (a.mean() - b.mean()) / np.sqrt(sp2 * (1 / n1 + 1 / n2))
print(f"154 two-sample t (pooled): t={t2:.3f}, df={n1+n2-2}")

# 155: Welch's t — неравные дисперсии
c = rng.normal(0.0, 1.0, size=30)
d = rng.normal(0.5, 3.0, size=30)
s1, s2 = c.var(ddof=1), d.var(ddof=1)
t_w = (c.mean() - d.mean()) / np.sqrt(s1 / 30 + s2 / 30)
df_w = (s1 / 30 + s2 / 30) ** 2 / ((s1 / 30) ** 2 / 29 + (s2 / 30) ** 2 / 29)
print(f"155 Welch t: t={t_w:.3f}, df≈{df_w:.1f}")

# 156: paired
before = rng.normal(70, 5, size=20)
after = before - rng.normal(2, 1.5, size=20)  # эффект "лечения"
diff = before - after
t_p = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))
print(f"156 paired t: t={t_p:.3f}, df={len(diff)-1}")

# 157: z-test (sigma известно)
sigma = 1.0
x_z = rng.normal(0.2, sigma, size=200)
z = (x_z.mean() - 0) / (sigma / np.sqrt(len(x_z)))
p_z = 2 * (1 - norm_cdf(abs(z)))
print(f"157 z-test: z={z:.3f}, p={p_z:.4f}")
