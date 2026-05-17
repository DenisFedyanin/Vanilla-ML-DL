"""Раздел 61 — Вероятность.

Файл 3: Неравенства концентрации.

Концепция: Неравенство Маркова.
Для X>=0 и a>0: P(X>=a) <= E[X]/a. Самое слабое, но требует только E[X].

Концепция: Неравенство Чебышева.
P(|X - mu| >= k*sigma) <= 1/k^2. Использует дисперсию.

Концепция: Неравенство Йенсена.
Для выпуклой g и СВ X: g(E[X]) <= E[g(X)]. Дает нижнюю грань на E[g(X)].
Применяется в EM, KL>=0, и других выводах.

Концепция: Неравенство Хоффдинга.
Для независимых X_i in [a_i, b_i] с E[X_i]=mu:
P(|sum X_i / n - mu| >= t) <= 2 exp(-2 n t^2 / (b-a)^2). Экспоненциальная концентрация.
"""
import numpy as np

rng = np.random.default_rng(0)
N = 50_000

# === Markov: X=Exp(1), a=3, теор. бунд = 1/3, истинная P = e^{-3}≈0.05 ===
X = rng.exponential(1.0, N)
true_p = (X >= 3).mean()
markov_bound = X.mean() / 3
print(f"Markov: истинная P(X>=3)={true_p:.4f}, бунд={markov_bound:.4f}")

# === Chebyshev: для N(0,1), k=2 ===
Z = rng.normal(0, 1, N)
true_p = (np.abs(Z) >= 2).mean()
cheb_bound = 1 / 4
print(f"Chebyshev (k=2): истинная P(|Z|>=2)={true_p:.4f}, бунд={cheb_bound:.4f}")

# === Jensen: g(x)=x^2, X~Uniform(-1,1). E[X]=0, E[X^2]=1/3 ===
U = rng.uniform(-1, 1, N)
print(f"Jensen: g(E[X])={(U.mean())**2:.4f} <= E[g(X)]={(U**2).mean():.4f}")

# === Hoeffding: X_i ~ Bernoulli(0.5), n=100, t=0.1 ===
n = 100; t = 0.1
trials = 5000
means = rng.binomial(1, 0.5, size=(trials, n)).mean(axis=1)
emp_tail = (np.abs(means - 0.5) >= t).mean()
hoef_bound = 2 * np.exp(-2 * n * t * t / 1.0)  # (b-a)^2 = 1
print(f"Hoeffding (n=100, t=0.1): эмп. tail={emp_tail:.4f}, бунд={hoef_bound:.4f}")

# Сравнение бундов для одного и того же события (нормальные X, |X-mu|>=2*sigma):
print("\nИтог: чем больше известно о распределении — тем точнее бунд:")
print(f"  Markov 1/k       (нужно только E)             = {1/2:.4f}")
print(f"  Chebyshev 1/k^2  (E и Var)                    = {1/4:.4f}")
print(f"  Hoeffding (нужно ограниченность) — на средние, экспоненциально мал")
