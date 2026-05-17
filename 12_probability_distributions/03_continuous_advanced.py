"""Раздел 12 — Распределения вероятностей.

Файл 3: продвинутые непрерывные (семейство сопряжённых для байеса).

Концепция 140: Beta(alpha, beta).
Распределение на [0,1]. Сопряжено к Bernoulli/Binomial: если априор Beta(a,b),
а данных k успехов из n, то апостериор Beta(a+k, b+n-k). База A/B-тестов.

Концепция 141: Gamma(k, theta).
Сумма k экспоненциальных = Gamma. Сопряжено к параметру скорости Пуассона.
Положительная случайная величина, гибкая форма.

Концепция 142: Chi-Squared(k).
Сумма k квадратов независимых стандартных нормальных. Появляется в тестах
дисперсии и chi^2-тестах согласия / независимости.

Концепция 143: Student-t(nu).
Распределение нормированного выборочного среднего при неизвестной дисперсии.
При nu->inf становится нормальным, при малом nu — тяжёлые хвосты.

Концепция 144: F(d1, d2).
Отношение двух Chi^2 / их степеней свободы. Используется в ANOVA и
F-тесте равенства дисперсий.

Концепция 145: Dirichlet(alpha).
Многомерное обобщение Beta. Сэмплирует вектор вероятностей (сумма =1).
Сопряжён к Multinomial (LDA, тематическое моделирование).

Концепция 146: Wishart (концепт).
Распределение случайных ковариационных матриц. Сопряжённый априор для
обратной ковариации многомерной нормали. Здесь — лишь упоминание формулы.

Концепция 147: Inverse-Gamma (концепт).
Если X ~ Gamma(k, theta), то 1/X ~ InvGamma. Сопряжённый априор для
дисперсии нормального распределения. Демонстрируем через 1/Gamma.
"""
import numpy as np

rng = np.random.default_rng(0)
N = 100_000

# 140: Beta(2,5) — скошено к 0
b = rng.beta(a=2.0, b=5.0, size=N)
print(f"140 Beta(2,5): mean={b.mean():.3f} (теор=a/(a+b)=0.286)")

# 141: Gamma(k=3, theta=2)
g = rng.gamma(shape=3.0, scale=2.0, size=N)
print(f"141 Gamma(3,2): mean={g.mean():.2f} (теор=k*theta=6), var={g.var():.2f} (теор=12)")

# 142: Chi^2(4) = sum of 4 N(0,1)^2
z = rng.normal(size=(N, 4))
chi2 = (z ** 2).sum(axis=1)
print(f"142 Chi^2(4): mean={chi2.mean():.2f} (теор=k=4), var={chi2.var():.2f} (теор=2k=8)")

# 143: Student-t(5)
t = rng.standard_t(df=5, size=N)
print(f"143 Student-t(5): mean={t.mean():.3f}, var={t.var():.2f} (теор=nu/(nu-2)≈1.67)")

# 144: F(5, 10) через два Chi^2
chi_a = (rng.normal(size=(N, 5)) ** 2).sum(axis=1) / 5
chi_b = (rng.normal(size=(N, 10)) ** 2).sum(axis=1) / 10
F = chi_a / chi_b
print(f"144 F(5,10): median={np.median(F):.3f} (теор медианы ~0.93), теор mean=d2/(d2-2)=1.25")

# 145: Dirichlet — простой 3-симплекс
d = rng.dirichlet(alpha=[2.0, 3.0, 5.0], size=N)
print(f"145 Dirichlet(2,3,5): средние = {d.mean(axis=0).round(3)} (теор a_i/sum=[0.2,0.3,0.5])")

# 146: Wishart (концепт) — оценим E[W] = nu * V для V=I, nu=5 через моделирование
nu, p = 5, 2
V = np.eye(p)
W_samples = []
for _ in range(2000):
    X = rng.multivariate_normal(np.zeros(p), V, size=nu)
    W_samples.append(X.T @ X)
W_mean = np.mean(W_samples, axis=0)
print(f"146 Wishart(nu=5, V=I): E[W]≈\n{W_mean.round(2)} (теор=nu*V=5*I)")

# 147: Inverse-Gamma через 1/Gamma. Если k=3, theta=2, то E[1/X]=1/(theta*(k-1))=1/4
inv_g = 1.0 / rng.gamma(shape=3.0, scale=2.0, size=N)
print(f"147 InvGamma(3,2): mean={inv_g.mean():.3f} (теор=1/(theta*(k-1))=0.25)")
