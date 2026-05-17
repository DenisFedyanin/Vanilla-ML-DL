"""Раздел 13 — Статистические тесты.

Файл 3: непараметрические тесты (без предположения о нормальности).

Концепция 162: Mann-Whitney U.
Объединяем две выборки, ранжируем. U1 = сумма рангов первой - n1(n1+1)/2.
Тест "одна выборка систематически больше другой". Заменитель t-теста.

Концепция 163: Wilcoxon signed-rank.
Парный аналог. Считаем разности, берём ранги |d|, суммируем со знаком d.
Под H0 (нет эффекта) ~ симметричное распределение вокруг 0.

Концепция 164: Kruskal-Wallis (концепт).
ANOVA для рангов: ранжируем все наблюдения, считаем H по средним рангам
в группах. Распределение под H0 ~ chi^2(k-1).

Концепция 165: Знаковый тест (sign test).
Очень простой парный тест: считаем, в скольких парах x_i > y_i. Под H0
это Bernoulli(0.5), т.е. Binomial. Не использует величин разностей.

Концепция 166: Тест Фридмана (концепт).
Аналог повторных измерений ANOVA на рангах внутри блока (subject). Ранжируем
условия для каждого subject отдельно, сравниваем средние ранги.
"""
import numpy as np

rng = np.random.default_rng(0)

# 162: Mann-Whitney U
a = rng.normal(0, 1, size=30)
b = rng.normal(0.7, 1, size=30)
all_vals = np.concatenate([a, b])
ranks = np.empty_like(all_vals)
ranks[np.argsort(all_vals)] = np.arange(1, len(all_vals) + 1)
R1 = ranks[: len(a)].sum()
n1, n2 = len(a), len(b)
U1 = R1 - n1 * (n1 + 1) / 2
U2 = n1 * n2 - U1
print(f"162 Mann-Whitney: U1={U1:.0f}, U2={U2:.0f} (min={min(U1,U2):.0f})")

# 163: Wilcoxon signed-rank
before = rng.normal(50, 10, size=20)
after = before - rng.normal(3, 5, size=20)
d = before - after
d = d[d != 0]
ranks_abs = np.empty_like(d)
ranks_abs[np.argsort(np.abs(d))] = np.arange(1, len(d) + 1)
W_pos = ranks_abs[d > 0].sum()
W_neg = ranks_abs[d < 0].sum()
print(f"163 Wilcoxon signed-rank: W+={W_pos:.0f}, W-={W_neg:.0f}")

# 164: Kruskal-Wallis (концепт)
g1 = rng.normal(0, 1, size=20)
g2 = rng.normal(0.5, 1, size=20)
g3 = rng.normal(1.0, 1, size=20)
all_kw = np.concatenate([g1, g2, g3])
r_kw = np.empty_like(all_kw)
r_kw[np.argsort(all_kw)] = np.arange(1, len(all_kw) + 1)
N = len(all_kw)
sizes = [20, 20, 20]
offsets = np.cumsum([0] + sizes)
sum_r = [r_kw[offsets[i]:offsets[i + 1]].sum() for i in range(3)]
H = 12 / (N * (N + 1)) * sum((s ** 2) / sz for s, sz in zip(sum_r, sizes)) - 3 * (N + 1)
print(f"164 Kruskal-Wallis H={H:.3f}, df=2")

# 165: sign test
x = rng.normal(0, 1, size=25)
y = x - rng.normal(0.4, 1, size=25)
n_pos = int((x > y).sum())
n_total = int((x != y).sum())
print(f"165 sign test: positives={n_pos}/{n_total} (под H0 ~ Binom(n, 0.5))")

# 166: Friedman (концепт) — 3 условия, 10 subjects
data = rng.normal(size=(10, 3)) + np.array([0, 0.3, 0.6])
# ранги внутри строки
ranks_f = np.argsort(np.argsort(data, axis=1), axis=1) + 1
R_j = ranks_f.sum(axis=0)
k_cond = 3
n_sub = 10
chi2_F = 12 / (n_sub * k_cond * (k_cond + 1)) * (R_j ** 2).sum() - 3 * n_sub * (k_cond + 1)
print(f"166 Friedman chi2={chi2_F:.3f}, df={k_cond-1}, ранг суммы={R_j}")
