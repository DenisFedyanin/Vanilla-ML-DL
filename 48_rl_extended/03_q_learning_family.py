"""Раздел 48 — RL расширенный.

Файл 3: семейство Q-learning.

Концепция: SARSA.
Q(s,a) <- Q(s,a) + alpha [r + gamma Q(s', a') - Q(s,a)]. On-policy:
обновление использует a', реально выбранное политикой (eps-greedy).

Концепция: Expected SARSA.
Вместо Q(s', a') берём E_{a'~pi}[Q(s', a')]. Меньше дисперсия, чем у SARSA;
работает и on-, и off-policy.

Концепция: Q-learning.
Q(s,a) <- Q(s,a) + alpha [r + gamma max_{a'} Q(s', a') - Q(s,a)]. Off-policy:
учим оптимальную политику независимо от действий поведения.

Концепция: Double Q-learning.
Двойная Q (A и B); один выбирает argmax, другой оценивает. Снимает
overestimation bias классического Q-learning (E[max] >= max E).
"""
import numpy as np

rng = np.random.default_rng(0)
# GridWorld 4x4, цель (3,3), R=+1; шаги R=-0.05
H, W = 4, 4
n_states = H * W
n_actions = 4
gamma = 0.95
alpha = 0.1
eps = 0.1


def step(s, a):
    i, j = s // W, s % W
    ds = [(-1, 0), (1, 0), (0, -1), (0, 1)][a]
    ni, nj = i + ds[0], j + ds[1]
    if not (0 <= ni < H and 0 <= nj < W):
        ni, nj = i, j
    sp = ni * W + nj
    if sp == n_states - 1:
        return sp, 1.0, True
    return sp, -0.05, False


def eps_greedy(Q, s, eps):
    if rng.uniform() < eps:
        return rng.integers(0, n_actions)
    return int(np.argmax(Q[s]))


def run(method="qlearn", episodes=500):
    Q = np.zeros((n_states, n_actions))
    if method == "double":
        Q2 = np.zeros((n_states, n_actions))
    returns = []
    for _ in range(episodes):
        s = 0
        ep_ret = 0
        done = False
        a = eps_greedy(Q, s, eps)
        for _ in range(100):
            sp, r, done = step(s, a)
            ep_ret += r
            ap = eps_greedy(Q, sp, eps)
            if method == "sarsa":
                target = r + (0 if done else gamma * Q[sp, ap])
                Q[s, a] += alpha * (target - Q[s, a])
            elif method == "expected":
                # eps-greedy expectation
                best = int(np.argmax(Q[sp]))
                probs = np.full(n_actions, eps / n_actions)
                probs[best] += 1 - eps
                target = r + (0 if done else gamma * (probs * Q[sp]).sum())
                Q[s, a] += alpha * (target - Q[s, a])
            elif method == "qlearn":
                target = r + (0 if done else gamma * Q[sp].max())
                Q[s, a] += alpha * (target - Q[s, a])
            elif method == "double":
                if rng.uniform() < 0.5:
                    a_max = int(np.argmax(Q[sp]))
                    target = r + (0 if done else gamma * Q2[sp, a_max])
                    Q[s, a] += alpha * (target - Q[s, a])
                else:
                    a_max = int(np.argmax(Q2[sp]))
                    target = r + (0 if done else gamma * Q[sp, a_max])
                    Q2[s, a] += alpha * (target - Q2[s, a])
            s, a = sp, ap
            if done:
                break
        returns.append(ep_ret)
    return Q, returns


for m in ("sarsa", "expected", "qlearn", "double"):
    Q, rets = run(m, 600)
    v = Q.max(axis=1).reshape(H, W)
    print(f"{m:9s} | финальный средний return (последние 50) = {np.mean(rets[-50:]):.2f} "
          f"| V[(0,0)] = {v[0, 0]:.2f}, V[(2,2)] = {v[2, 2]:.2f}")

print("\nВсе методы сходятся к похожей V, но Q-learning может переоценивать,")
print("Double Q убирает оптимизм; Expected SARSA — самая стабильная среди on-policy.")
