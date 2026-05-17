"""Раздел 55 — MCMC. Файл 2: Metropolis-Hastings и Gibbs.

Концепция: Markov Chain Monte Carlo.
Строим цепь Маркова, стационарное распределение которой — наш target p(x).
После burn-in сэмплы из цепи ~ p(x). Не нужно знать нормировку p (только
до константы) — главное преимущество.

Концепция: Random-walk Metropolis-Hastings.
1) x' = x + N(0, sigma) (симметричное предложение).
2) alpha = min(1, p(x') / p(x)).
3) С вероятностью alpha принимаем x', иначе остаёмся в x.
Сходится к p при выполнении detailed balance.

Концепция: Выбор шага.
Слишком маленький sigma — медленное движение по пространству, цепь не
исследует все области. Слишком большой — много отказов. Оптимальный
acceptance rate ≈ 0.234 для многомерных гладких target.

Концепция: Gibbs sampling.
Для p(x, y): поочерёдно сэмплируем x ~ p(x | y), y ~ p(y | x). Каждый шаг
условный — часто доступен в закрытой форме. Принятие всегда (acceptance = 1).
Подходит, когда условные распределения легко взять.

Концепция: Диагностика сходимости.
Trace plot, ESS, Gelman-Rubin R-hat (несколько цепей, отношение between/within
variance). R-hat близкий к 1 — сходимость. Скачки на trace plot — плохо.
"""
import numpy as np

rng = np.random.default_rng(0)


# === Random-walk MH на бимодальном target ===
def log_target(x):
    return np.log(0.4 * np.exp(-0.5 * (x + 2) ** 2) + 0.6 * np.exp(-0.5 * (x - 2) ** 2) + 1e-30)


N = 20000
x = 0.0
samples = np.zeros(N)
acc = 0
sigma = 2.0
for t in range(N):
    x_new = x + rng.normal(scale=sigma)
    if np.log(rng.uniform()) < log_target(x_new) - log_target(x):
        x = x_new
        acc += 1
    samples[t] = x

burn = 2000
print(f"MH bimodal: acceptance={acc / N:.3f}, mean={samples[burn:].mean():+.3f}  (теор=0.8)")
left = (samples[burn:] < 0).mean()
right = (samples[burn:] > 0).mean()
print(f"  доля сэмплов слева/справа от 0: {left:.2f} / {right:.2f}  (теор 0.4/0.6)")


# === Gibbs sampling для двумерного нормального ===
# Совместное N(mu, Sigma) с rho. Условные:
# x | y ~ N(rho*y, 1-rho^2),  y | x ~ N(rho*x, 1-rho^2)  (при mu=0, sigma=1)
rho = 0.8
x, y = 0.0, 0.0
N = 10000
xs = np.zeros(N); ys = np.zeros(N)
for t in range(N):
    x = rng.normal(loc=rho * y, scale=np.sqrt(1 - rho ** 2))
    y = rng.normal(loc=rho * x, scale=np.sqrt(1 - rho ** 2))
    xs[t] = x; ys[t] = y

print(f"\nGibbs bivariate normal (rho={rho}):")
print(f"  mean=({xs[1000:].mean():+.3f}, {ys[1000:].mean():+.3f})  (теор 0, 0)")
print(f"  std =({xs[1000:].std():.3f}, {ys[1000:].std():.3f})    (теор 1, 1)")
print(f"  corr={np.corrcoef(xs[1000:], ys[1000:])[0,1]:.3f}    (теор {rho})")
