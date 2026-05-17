"""Раздел 54 — Bayesian ML. Файл 2: Сопряжённые priors.

Концепция: Beta-Binomial.
Likelihood: Binomial(n, theta). Prior: Beta(a, b).
Posterior: Beta(a + k, b + n - k). Применение: A/B-тесты, CTR, доля брака.

Концепция: Gamma-Poisson.
Likelihood: Poisson(lambda), n наблюдений с суммой sum_x.
Prior: Gamma(alpha, beta) (shape, rate). Posterior: Gamma(alpha + sum_x, beta + n).
Применение: count-данные — звонки, клики, дефекты.

Концепция: Normal-Normal (известная sigma^2).
Likelihood: N(mu, sigma^2), n наблюдений x_i. Prior: mu ~ N(mu_0, tau_0^2).
Posterior:
  tau_n^2 = 1 / (1/tau_0^2 + n/sigma^2)
  mu_n    = tau_n^2 * (mu_0/tau_0^2 + sum_x/sigma^2)
Posterior mean = взвешенное среднее prior mean и выборочного среднего.

Концепция: Normal-Inverse-Gamma (неизвестные mu и sigma^2).
Совместный prior: sigma^2 ~ Inv-Gamma(a, b), mu | sigma^2 ~ N(mu_0, sigma^2/kappa_0).
Posterior — из того же семейства, формулы апдейта известны. Это полная байесовская
модель нормального распределения.

Концепция: Зачем conjugate.
Закрытые формулы — нет MCMC, нет численного интегрирования. Когда conjugate
не существует — приходится прибегать к VI или MCMC (главы 55).
"""
import numpy as np

rng = np.random.default_rng(0)

# === Beta-Binomial ===
a, b = 2.0, 2.0
n, k = 12, 8
print(f"Beta-Binomial: prior Beta({a},{b}), k={k}/{n}  ->  Beta({a+k},{b+n-k})")
print(f"  Posterior mean = {(a+k)/(a+k+b+n-k):.3f}")

# === Gamma-Poisson ===
alpha, beta = 2.0, 1.0       # prior на rate lambda
x = rng.poisson(lam=3.5, size=20)
alpha_n = alpha + x.sum()
beta_n = beta + len(x)
print(f"\nGamma-Poisson: prior Gamma({alpha},{beta}), sum_x={x.sum()}, n={len(x)}")
print(f"  Posterior Gamma({alpha_n}, {beta_n}), mean = {alpha_n/beta_n:.3f}  (истинный lambda=3.5)")

# === Normal-Normal (известная sigma) ===
mu_true = 5.0
sigma = 1.0
x = rng.normal(mu_true, sigma, size=15)
mu_0, tau_0 = 0.0, 10.0
tau_n2 = 1.0 / (1.0 / tau_0 ** 2 + len(x) / sigma ** 2)
mu_n = tau_n2 * (mu_0 / tau_0 ** 2 + x.sum() / sigma ** 2)
print(f"\nNormal-Normal: x_bar={x.mean():.3f}, mu_0={mu_0}")
print(f"  Posterior N({mu_n:.3f}, {tau_n2:.4f})  истинная mu={mu_true}")

# === Normal-Inverse-Gamma (неизвестные mu и sigma^2) ===
# Prior hyperparams
mu0, kappa0, a0, b0 = 0.0, 1.0, 2.0, 2.0
n = len(x)
xbar = x.mean()
S = ((x - xbar) ** 2).sum()
kappa_n = kappa0 + n
mu_n = (kappa0 * mu0 + n * xbar) / kappa_n
a_n = a0 + n / 2
b_n = b0 + 0.5 * S + 0.5 * kappa0 * n * (xbar - mu0) ** 2 / kappa_n
print(f"\nNormal-Inv-Gamma posterior:")
print(f"  mu_n={mu_n:.3f}, kappa_n={kappa_n}, a_n={a_n}, b_n={b_n:.3f}")
print(f"  E[sigma^2 | x] = b_n/(a_n - 1) = {b_n/(a_n-1):.3f}  истинная sigma^2={sigma**2}")
