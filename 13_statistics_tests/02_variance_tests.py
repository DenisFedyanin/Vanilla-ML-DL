"""Раздел 13 — Статистические тесты.

Файл 2: тесты на сравнение ДИСПЕРСИЙ и средних в нескольких группах.

Концепция 158: F-тест отношения дисперсий.
F = s1^2 / s2^2 при равных дисперсиях имеет F(n1-1, n2-1). Очень чувствителен
к ненормальности — используется только при близком к нормальному распределении.

Концепция 159: Тест Левена.
Заменяем x_ij на |x_ij - median_i| (или mean_i) и делаем ANOVA на этих
величинах. Гораздо робастнее F-теста и Бартлетта.

Концепция 160: Тест Бартлетта (концепт).
Похож на Левена, но строится из логарифмов выборочных дисперсий. Очень
чувствителен к отклонениям от нормальности — почему обычно предпочитают Левена.

Концепция 161: Однофакторная ANOVA.
Сравниваем средние k групп. Разбиваем общую сумму квадратов на
between-group (SSB) и within-group (SSW). F = (SSB/(k-1)) / (SSW/(N-k)).
"""
import numpy as np

rng = np.random.default_rng(0)

# 158: F-тест дисперсий
a = rng.normal(0, 1.0, size=40)
b = rng.normal(0, 2.0, size=40)  # дисперсия в 4 раза больше
F = a.var(ddof=1) / b.var(ddof=1)
print(f"158 F-test var: F={F:.3f} (теор ≈ 1/4=0.25)")

# 159: Levene (median-based, Brown-Forsythe)
groups = [rng.normal(0, 1, size=30), rng.normal(0, 2, size=30), rng.normal(0, 1.5, size=30)]
z_groups = [np.abs(g - np.median(g)) for g in groups]
all_z = np.concatenate(z_groups)
overall = all_z.mean()
N = len(all_z)
k = len(z_groups)
ssb = sum(len(z) * (z.mean() - overall) ** 2 for z in z_groups)
ssw = sum(((z - z.mean()) ** 2).sum() for z in z_groups)
W = (ssb / (k - 1)) / (ssw / (N - k))
print(f"159 Levene W={W:.3f}, df=({k-1},{N-k})")

# 160: Bartlett (концепт) — статистика
ni = np.array([len(g) for g in groups])
si2 = np.array([g.var(ddof=1) for g in groups])
sp2 = ((ni - 1) * si2).sum() / (N - k)
num = (N - k) * np.log(sp2) - ((ni - 1) * np.log(si2)).sum()
den = 1 + (1.0 / (3 * (k - 1))) * ((1 / (ni - 1)).sum() - 1 / (N - k))
chi2_bart = num / den
print(f"160 Bartlett chi2={chi2_bart:.3f}, df={k-1}")

# 161: one-way ANOVA
g1 = rng.normal(0.0, 1, size=30)
g2 = rng.normal(0.5, 1, size=30)
g3 = rng.normal(1.0, 1, size=30)
all_x = np.concatenate([g1, g2, g3])
grand = all_x.mean()
k = 3
ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in (g1, g2, g3))
ssw = sum(((g - g.mean()) ** 2).sum() for g in (g1, g2, g3))
F_anova = (ssb / (k - 1)) / (ssw / (len(all_x) - k))
print(f"161 one-way ANOVA F={F_anova:.3f}, df=({k-1},{len(all_x)-k})")
