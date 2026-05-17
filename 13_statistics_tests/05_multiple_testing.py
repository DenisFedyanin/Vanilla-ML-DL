"""Раздел 13 — Статистические тесты.

Файл 5: множественные сравнения, перестановочные тесты, McNemar.

Концепция 171: Поправка Бонферрони.
При m тестах используем alpha/m для каждого. Контролирует FWER, но крайне
консервативен. Простой и универсальный.

Концепция 172: Метод Холма.
Сортируем p-значения по возрастанию: p_(1) <= ... <= p_(m). Сравниваем
p_(i) с alpha/(m-i+1). Менее консервативен, чем Бонферрони, тоже FWER.

Концепция 173: Benjamini-Hochberg (FDR).
Сортируем p. Находим максимальное i, такое что p_(i) <= i/m * alpha.
Отвергаем H0 для всех p_(1)..p_(i). Контролирует false discovery rate
(долю ложных среди отвергнутых), а не FWER.

Концепция 174: Перестановочный тест.
Многократно перемешиваем метки групп и пересчитываем статистику. p-value =
доля перестановок с |stat_perm| >= |stat_obs|. Не предполагает распределения.

Концепция 175: Бутстрап-тест.
Сэмплируем с возвращением, оцениваем распределение статистики/доверительный
интервал. Используем для CI разности средних.

Концепция 176: Тест МакНемара.
Парный тест для бинарных данных (например, до/после или классификатор A vs B
на одной выборке). chi2 = (b-c)^2 / (b+c), где b и c — несогласия.
"""
import numpy as np

rng = np.random.default_rng(0)

pvals = np.array([0.001, 0.012, 0.03, 0.04, 0.2, 0.6])
m = len(pvals)
alpha = 0.05

# 171: Bonferroni
adj_bonf = np.minimum(pvals * m, 1.0)
print(f"171 Bonferroni adj: {adj_bonf.round(3)}, reject={adj_bonf < alpha}")

# 172: Holm
order = np.argsort(pvals)
holm = np.zeros_like(pvals)
running_max = 0.0
for rank, idx in enumerate(order):
    cand = pvals[idx] * (m - rank)
    running_max = max(running_max, cand)
    holm[idx] = min(running_max, 1.0)
print(f"172 Holm adj:       {holm.round(3)}, reject={holm < alpha}")

# 173: BH-FDR
order = np.argsort(pvals)
sorted_p = pvals[order]
bh = sorted_p * m / np.arange(1, m + 1)
# монотонизация сверху-вниз
for i in range(m - 2, -1, -1):
    bh[i] = min(bh[i], bh[i + 1])
bh_full = np.empty_like(bh)
bh_full[order] = np.minimum(bh, 1.0)
print(f"173 BH-FDR adj:     {bh_full.round(3)}, reject={bh_full < alpha}")

# 174: permutation test для разности средних
a = rng.normal(0, 1, size=40)
b = rng.normal(0.5, 1, size=40)
obs = a.mean() - b.mean()
pool = np.concatenate([a, b])
n_a = len(a)
B = 2000
count = 0
for _ in range(B):
    rng.shuffle(pool)
    stat = pool[:n_a].mean() - pool[n_a:].mean()
    if abs(stat) >= abs(obs):
        count += 1
p_perm = count / B
print(f"174 permutation test: obs diff={obs:.3f}, p={p_perm:.4f}")

# 175: bootstrap CI для среднего
x = rng.normal(0, 1, size=100)
B = 2000
boot_means = np.array([rng.choice(x, size=len(x), replace=True).mean() for _ in range(B)])
ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
print(f"175 bootstrap 95% CI mean: [{ci_low:.3f}, {ci_high:.3f}]")

# 176: McNemar — два классификатора на 100 объектах (по парам результатов)
# таблица: A_right/A_wrong x B_right/B_wrong
tab = np.array([[40, 15],
                [5, 40]])
b_, c_ = tab[0, 1], tab[1, 0]
mcnemar = (b_ - c_) ** 2 / (b_ + c_)
print(f"176 McNemar chi2={mcnemar:.3f}, b={b_}, c={c_}")
