"""Раздел 67 — Online learning.

Файл 2: Multi-armed bandits.

Концепция: Multi-armed bandit (MAB).
K 'рук' игрового автомата, каждая даёт случайную награду из неизвестного распределения.
Цель — максимизировать суммарную награду за T шагов. Trade-off: exploration vs exploitation.

Концепция: Epsilon-greedy.
С вероятностью eps пробуем случайную руку (exploration), с 1-eps берём руку с лучшим
текущим средним (exploitation). Простой, но требует подбора eps; иногда используют
убывающее eps_t = min(1, c*K/t).

Концепция: UCB1 (Auer 2002).
Берём руку с максимальным верхним доверительным интервалом:
score_i = mean_i + sqrt(2 * ln(t) / n_i), где n_i — сколько раз дёргали руку i.
Гарантированная regret O(sqrt(K T log T)). Никакой случайности.

Концепция: Thompson sampling (Bernoulli arms).
Bayesian подход: для каждой руки храним Beta(alpha, beta) посериор по награде.
На каждом шаге: семплируем theta_i ~ Beta(alpha_i, beta_i) для каждой руки,
дёргаем ту, у которой theta больше. Награда 1 -> alpha++, 0 -> beta++.

Концепция: Contextual bandits (LinUCB).
Кроме руки видим вектор контекста x_t (фичи юзера). Награда linear:
E[r | a, x] = x^T theta_a. На каждом шаге для каждой руки оцениваем theta_a
(ridge), считаем UCB: x^T theta_a + alpha * sqrt(x^T A_a^{-1} x). Выбираем максимум.
"""
import numpy as np

rng = np.random.default_rng(0)
K = 10
true_means = rng.uniform(0.1, 0.9, size=K)  # вероятности успеха каждой руки
best_arm = int(np.argmax(true_means))
T = 2000
print(f"Истинные вероятности рук (best={best_arm}, p={true_means[best_arm]:.2f}):")
print(np.round(true_means, 2))

def pull(arm):
    return float(rng.uniform() < true_means[arm])

# === Epsilon-greedy ===
counts = np.zeros(K)
means = np.zeros(K)
total = 0.0
eps = 0.1
for t in range(T):
    if rng.uniform() < eps or counts.min() == 0:
        a = int(rng.integers(0, K))
    else:
        a = int(np.argmax(means))
    r = pull(a)
    counts[a] += 1
    means[a] += (r - means[a]) / counts[a]
    total += r
print(f"eps-greedy: reward={total:.0f}, best arm pulls={counts[best_arm]:.0f}/{T}")

# === UCB1 ===
counts = np.zeros(K)
means = np.zeros(K)
total_ucb = 0.0
# обязательно дёрнуть каждую руку 1 раз
for a in range(K):
    r = pull(a)
    counts[a] += 1
    means[a] = r
    total_ucb += r
for t in range(K, T):
    ucb = means + np.sqrt(2 * np.log(t) / counts)
    a = int(np.argmax(ucb))
    r = pull(a)
    counts[a] += 1
    means[a] += (r - means[a]) / counts[a]
    total_ucb += r
print(f"UCB1      : reward={total_ucb:.0f}, best arm pulls={counts[best_arm]:.0f}/{T}")

# === Thompson Sampling (Beta-Bernoulli) ===
alpha = np.ones(K)
beta = np.ones(K)
total_ts = 0.0
counts_ts = np.zeros(K)
for t in range(T):
    samples = rng.beta(alpha, beta)
    a = int(np.argmax(samples))
    r = pull(a)
    alpha[a] += r
    beta[a] += 1 - r
    counts_ts[a] += 1
    total_ts += r
print(f"Thompson  : reward={total_ts:.0f}, best arm pulls={counts_ts[best_arm]:.0f}/{T}")

# === LinUCB (концепт + минимальный демо) ===
# 2 руки с контекстом 3D. Истинные параметры:
d = 3
theta_true = np.array([[1.0, -1.0, 0.5], [-0.5, 1.0, 0.2]])
A = [np.eye(d) for _ in range(2)]
b = [np.zeros(d) for _ in range(2)]
alpha_lin = 1.0
total_lin = 0.0
for t in range(500):
    x = rng.normal(size=d)
    scores = []
    for a in range(2):
        Ainv = np.linalg.inv(A[a])
        theta_hat = Ainv @ b[a]
        ucb = x @ theta_hat + alpha_lin * np.sqrt(x @ Ainv @ x)
        scores.append(ucb)
    a = int(np.argmax(scores))
    r = x @ theta_true[a] + 0.1 * rng.normal()
    A[a] += np.outer(x, x)
    b[a] += r * x
    total_lin += r
print(f"LinUCB    : суммарная награда за 500 шагов = {total_lin:.1f}")
