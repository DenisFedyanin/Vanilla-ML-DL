"""Раздел 67 — Online learning.

Файл 3: Regret и трейдофф exploration/exploitation.

Концепция: Regret.
Regret_T = T * mu* - sum_{t=1..T} mu_{a_t} = ожидаемое 'недополученное' по сравнению с
оракулом, который всегда дёргает лучшую руку. Чем медленнее растёт regret, тем лучше
алгоритм. Идеал — sublinear regret (например O(sqrt(T)) или O(log T)).

Концепция: Cumulative regret plot.
Если regret растёт линейно — алгоритм 'не учится'. Если как sqrt(T) или log(T) —
учится, лишь иногда экспериментируя. Random strategy всегда линеен.

Концепция: Exploration vs Exploitation.
Чисто exploit — рискуем застрять на субоптимальной руке (если повезло вначале).
Чисто explore — теряем награду, дёргая плохие руки.
UCB и Thompson балансируют это автоматически, eps-greedy — через явный параметр.

Концепция: Theoretical bounds.
- UCB1: regret = O(sum_{a: mu_a < mu*} log(T) / (mu* - mu_a)). Логарифмический в T.
- Thompson sampling (Bernoulli): почти оптимальный, тоже O(log T).
- Random: linear O(T).

Демо: запускаем eps-greedy, UCB1, Thompson, random на 10-armed bandit и печатаем
кумулятивный regret в нескольких точках.
"""
import numpy as np

rng = np.random.default_rng(0)
K = 10
true_means = rng.uniform(0.1, 0.9, size=K)
mu_star = true_means.max()
T = 5000

def pull(a):
    return float(rng.uniform() < true_means[a])

def run(strategy):
    if strategy == "epsg":
        eps = 0.1
    counts = np.zeros(K)
    means = np.zeros(K)
    alpha = np.ones(K)
    beta = np.ones(K)
    regret = np.zeros(T)
    cum_r = 0.0
    for t in range(T):
        if strategy == "random":
            a = int(rng.integers(0, K))
        elif strategy == "epsg":
            if rng.uniform() < eps or counts.min() == 0:
                a = int(rng.integers(0, K))
            else:
                a = int(np.argmax(means))
        elif strategy == "ucb":
            if counts.min() == 0:
                a = int(np.argmin(counts))
            else:
                ucb = means + np.sqrt(2 * np.log(t + 1) / counts)
                a = int(np.argmax(ucb))
        elif strategy == "ts":
            a = int(np.argmax(rng.beta(alpha, beta)))
        r = pull(a)
        counts[a] += 1
        means[a] += (r - means[a]) / counts[a]
        alpha[a] += r
        beta[a] += 1 - r
        cum_r += mu_star - true_means[a]
        regret[t] = cum_r
    return regret

reg_random = run("random")
reg_eps = run("epsg")
reg_ucb = run("ucb")
reg_ts = run("ts")

print(f"True means: best mu*={mu_star:.3f}, K={K}, T={T}")
print(f"{'t':>5} {'random':>8} {'epsg':>8} {'UCB1':>8} {'Thompson':>9}")
for t in [50, 200, 500, 1000, 2000, T - 1]:
    print(f"{t+1:>5} {reg_random[t]:>8.1f} {reg_eps[t]:>8.1f} {reg_ucb[t]:>8.1f} {reg_ts[t]:>9.1f}")

print("\nИнтерпретация:")
print("  random — растёт линейно (нет обучения).")
print("  eps-greedy — также с линейной частью из-за постоянного eps.")
print("  UCB1 / Thompson — растут заметно медленнее (логарифмически).")
print("\nExploration-exploitation: дилемма выбора между 'попробовать новое' и 'эксплуатировать лучшее'.")
