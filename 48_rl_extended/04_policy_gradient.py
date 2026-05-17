"""Раздел 48 — RL расширенный.

Файл 4: Policy Gradient методы.

Концепция: REINFORCE (Monte-Carlo PG).
grad J = E[grad log pi(a|s) * G_t]. Обновление: theta += alpha * grad log pi * G.
Высокая дисперсия, но не нужны value-функции.

Концепция: REINFORCE with baseline.
Заменяем G_t -> (G_t - b(s)). b(s) — обычно V(s). Уменьшает дисперсию, не
вводит смещение. Стандартный приём.

Концепция: Actor-Critic.
Actor pi(a|s; theta) выбирает действия; Critic V(s; w) оценивает. Используем
TD-error delta = r + gamma V(s') - V(s) как advantage. Обновляем оба online.

Концепция: A2C / A3C.
Advantage actor-critic с явным A(s,a) = Q(s,a) - V(s). A3C — асинхронные
параллельные акторы (исторически), A2C — синхронная версия.

Концепция: GAE (Generalized Advantage Estimation).
A_t = sum_l (gamma * lambda)^l * delta_{t+l}. Гладкая интерполяция между
n-step TD и MC.
"""
import numpy as np

rng = np.random.default_rng(0)
# Multi-armed bandit с K руками
K = 4
true_rewards = np.array([0.1, 0.3, 0.7, 0.2])


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# REINFORCE на бандите (одно "состояние")
theta = np.zeros(K)
lr = 0.05
for t in range(1000):
    p = softmax(theta)
    a = rng.choice(K, p=p)
    r = true_rewards[a] + 0.05 * rng.normal()
    # grad log pi(a) = e_a - p
    grad = -p
    grad[a] += 1
    theta += lr * r * grad
print(f"REINFORCE: финальные probs = {softmax(theta).round(2)}, лучшая рука = 2 (true=0.7)")

# REINFORCE with baseline (average return)
theta = np.zeros(K)
baseline = 0.0
for t in range(1000):
    p = softmax(theta)
    a = rng.choice(K, p=p)
    r = true_rewards[a] + 0.05 * rng.normal()
    advantage = r - baseline
    grad = -p
    grad[a] += 1
    theta += lr * advantage * grad
    baseline += 0.01 * (r - baseline)
print(f"REINFORCE+baseline: probs = {softmax(theta).round(2)}, baseline = {baseline:.3f}")

# Actor-Critic на маленьком GridWorld 3x3 (цель угол)
H = W = 3
n_states = 9
n_actions = 4
gamma = 0.9


def step(s, a):
    i, j = s // W, s % W
    di, dj = [(-1, 0), (1, 0), (0, -1), (0, 1)][a]
    ni, nj = max(0, min(H - 1, i + di)), max(0, min(W - 1, j + dj))
    sp = ni * W + nj
    done = sp == n_states - 1
    r = 1.0 if done else -0.04
    return sp, r, done


theta_ac = np.zeros((n_states, n_actions))
V = np.zeros(n_states)
lr_a, lr_c = 0.05, 0.1
for ep in range(500):
    s = 0
    for _ in range(50):
        p = softmax(theta_ac[s])
        a = rng.choice(n_actions, p=p)
        sp, r, done = step(s, a)
        target = r + (0 if done else gamma * V[sp])
        delta = target - V[s]
        V[s] += lr_c * delta
        grad = -p
        grad[a] += 1
        theta_ac[s] += lr_a * delta * grad
        if done:
            break
        s = sp

print("\nActor-Critic V (3x3):")
print(V.reshape(H, W).round(2))
print("Жадная политика:")
arrows = ["U", "D", "L", "R"]
for i in range(H):
    print([arrows[int(np.argmax(theta_ac[i * W + j]))] for j in range(W)])

# GAE формула
deltas = np.array([0.5, -0.1, 0.3, 0.8])
gamma_v, lam = 0.99, 0.95
A_gae = np.zeros_like(deltas)
gae = 0
for t in reversed(range(len(deltas))):
    gae = deltas[t] + gamma_v * lam * gae
    A_gae[t] = gae
print(f"\nGAE advantages: {A_gae.round(3)} (lambda={lam})")
