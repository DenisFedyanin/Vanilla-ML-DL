"""Раздел 54 — Bayesian ML. Файл 1: Prior / Likelihood / Posterior / Evidence.

Концепция: Теорема Байеса.
p(theta | D) = p(D | theta) * p(theta) / p(D).
- p(theta) — априорное распределение (что мы знали до данных)
- p(D | theta) — правдоподобие (как данные зависят от параметра)
- p(theta | D) — апостериорное (что знаем после данных)
- p(D) = ∫ p(D|theta) p(theta) dtheta — evidence (нормировка).

Концепция: Bayes для монеты с Beta-prior.
theta = вероятность выпадения 'герба'. Если prior = Beta(a, b), данные = k гербов
из n бросков, тогда posterior = Beta(a + k, b + n - k). Это classic conjugate.

Концепция: Сопряжённое (conjugate) prior.
Семейство prior, при котором posterior — из того же семейства. Сильно упрощает
вычисления: формула для апдейта параметров вместо численного интеграла.

Концепция: Posterior mean vs MAP.
Posterior mean = E[theta | D] — точка минимума ожидаемого квадратичного риска.
MAP = argmax p(theta | D) — точка минимума 0/1-loss. Для симметричных posterior
они совпадают; для скошенных — нет.

Концепция: Credible interval.
[q_0.025, q_0.975] от posterior — байесовский 95% credible interval.
В отличие от частотного confidence interval, имеет прямую интерпретацию
'theta лежит здесь с вероятностью 95%'.
"""
import numpy as np


def beta_logpdf(x, a, b):
    # log Beta(x; a, b) = (a-1)log x + (b-1)log(1-x) - log B(a,b)
    # B(a,b) посчитаем через log-gamma вручную через Стирлинга? Лучше через рекуррентность.
    # Используем простой подход: ненормированная плотность для построения posterior.
    return (a - 1) * np.log(np.clip(x, 1e-12, 1)) + (b - 1) * np.log(np.clip(1 - x, 1e-12, 1))


# Данные: 8 гербов из 12 бросков. Prior Beta(2, 2) — слабое предпочтение к 0.5.
n_data, k = 12, 8
a_prior, b_prior = 2.0, 2.0
a_post, b_post = a_prior + k, b_prior + (n_data - k)
print(f"Prior Beta({a_prior}, {b_prior}), данные: {k}/{n_data}")
print(f"Posterior Beta({a_post}, {b_post})")

# Mean = a / (a+b), Var = ab / ((a+b)^2 (a+b+1))
m_prior = a_prior / (a_prior + b_prior)
m_post = a_post / (a_post + b_post)
v_post = a_post * b_post / ((a_post + b_post) ** 2 * (a_post + b_post + 1))
print(f"E[theta] prior = {m_prior:.3f}, posterior = {m_post:.3f}")
print(f"MLE (просто k/n) = {k / n_data:.3f}")
print(f"Var(posterior)   = {v_post:.4f}, sd = {np.sqrt(v_post):.3f}")

# Credible interval численно: через сетку
grid = np.linspace(0.001, 0.999, 2000)
log_p = beta_logpdf(grid, a_post, b_post)
p = np.exp(log_p - log_p.max())
p = p / p.sum()
cdf = np.cumsum(p)
ci_low = grid[np.searchsorted(cdf, 0.025)]
ci_high = grid[np.searchsorted(cdf, 0.975)]
print(f"95% credible interval = [{ci_low:.3f}, {ci_high:.3f}]")

# MAP: (a-1)/(a+b-2) для Beta при a,b>1
map_ = (a_post - 1) / (a_post + b_post - 2)
print(f"MAP theta = {map_:.3f}")

# Evidence = p(D) = integral. Для conjugate Beta-Binomial:
# p(D) = C(n,k) * B(a+k, b+n-k) / B(a, b). Покажем как сетка.
prior = np.exp(beta_logpdf(grid, a_prior, b_prior))
dx = grid[1] - grid[0]
prior = prior / (prior.sum() * dx)
like = grid ** k * (1 - grid) ** (n_data - k)
evidence = (like * prior).sum() * dx
print(f"Evidence p(D) ≈ {evidence:.4e}  (нормировка posterior)")
