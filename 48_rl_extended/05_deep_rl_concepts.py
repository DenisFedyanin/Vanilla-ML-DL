"""Раздел 48 — RL расширенный.

Файл 5: глубинные RL концепции и exploration.

Концепция: DQN (Deep Q-Network).
Q(s,a; theta) — нейросеть. Replay buffer хранит (s,a,r,s'); target network
theta^- обновляется реже. Loss = (r + gamma max_a' Q_target(s',a') - Q(s,a))^2.

Концепция: Double DQN.
Target = r + gamma * Q_target(s', argmax_a' Q(s', a')). Online сеть выбирает
действие, target — оценивает. Уменьшает overestimation как в Double Q.

Концепция: Dueling DQN.
Q(s,a) = V(s) + (A(s,a) - mean_a A). Разделение V и advantage помогает в
состояниях, где выбор действия слабо влияет на возврат.

Концепция: Rainbow.
Объединение: Double + Dueling + PER (priority replay) + n-step + distributional
+ noisy nets. Сильно лучше базового DQN.

Концепция: PPO (Proximal Policy Optimization).
L_clip(theta) = E[min(r_t A_t, clip(r_t, 1-eps, 1+eps) A_t)], где r_t — ratio
pi_theta / pi_old. Не пускает большой шаг политики, без сложных trust-region.

Концепция: DDPG, TD3, SAC.
DDPG — детерминированный actor-critic для непрерывных действий. TD3 — двойной
critic, отложенный actor, target policy smoothing. SAC — максимизация энтропии.

Концепция: Exploration.
Epsilon-greedy — простой baseline. UCB1: a = argmax (Q + sqrt(2 log t / n_a)).
Thompson sampling: sample из posterior Beta(a, b) и argmax.
"""
import numpy as np
from collections import deque

rng = np.random.default_rng(0)

# Минимальный replay buffer
buf = deque(maxlen=1000)
for _ in range(100):
    buf.append((rng.normal(size=4), rng.integers(0, 2),
                rng.normal(), rng.normal(size=4), False))
print(f"Replay buffer: размер = {len(buf)}, ёмкость = 1000")
batch_idx = rng.choice(len(buf), size=8, replace=False)
batch = [buf[i] for i in batch_idx]
print(f"Sample batch: {len(batch)} переходов")

# Target network: концепт через soft update tau
theta_online = rng.normal(size=5)
theta_target = theta_online.copy()
# одно обновление online
theta_online += 0.1 * rng.normal(size=5)
tau = 0.005
theta_target = tau * theta_online + (1 - tau) * theta_target
print(f"Soft target update tau={tau}: max|diff| = {np.abs(theta_target - theta_online).max():.3f}")

# Dueling: Q = V + A - mean(A)
V_s = 1.5
A_sa = np.array([0.2, -0.3, 0.5, -0.1])
Q_dueling = V_s + A_sa - A_sa.mean()
print(f"Dueling Q(s,a) = {Q_dueling.round(2)} (sum=V*K+0)")

# PPO clip
def ppo_clip_obj(ratio, A, eps=0.2):
    return np.minimum(ratio * A, np.clip(ratio, 1 - eps, 1 + eps) * A)


ratio = np.array([0.5, 0.9, 1.0, 1.3, 1.6])
A = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
print(f"PPO clip (A=1): ratios {ratio} -> objective {ppo_clip_obj(ratio, A, 0.2)}")
print("Видно: при ratio>1.2 объектив зажимается; нет смысла расти дальше")

# Epsilon-greedy
Q_vals = np.array([1.0, 2.0, 0.5, 1.5])
eps = 0.1
choices = []
for _ in range(1000):
    if rng.uniform() < eps:
        choices.append(rng.integers(0, 4))
    else:
        choices.append(int(np.argmax(Q_vals)))
print(f"Epsilon-greedy (eps=0.1): доля жадных = {(np.array(choices) == 1).mean():.2f}")

# UCB1
K = 4
true_rewards = np.array([0.2, 0.5, 0.8, 0.4])
counts = np.zeros(K)
sums = np.zeros(K)
# Сначала по разу каждый
for a in range(K):
    counts[a] = 1
    sums[a] = true_rewards[a] + 0.05 * rng.normal()
for t in range(K, 500):
    ucb = sums / counts + np.sqrt(2 * np.log(t + 1) / counts)
    a = int(np.argmax(ucb))
    counts[a] += 1
    sums[a] += true_rewards[a] + 0.05 * rng.normal()
print(f"UCB1 (true best=2): pulls per arm = {counts.astype(int)}")

# Thompson sampling (Bernoulli с Beta prior)
alphas = np.ones(K)
betas = np.ones(K)
true_p = np.array([0.2, 0.5, 0.8, 0.4])
pulls = np.zeros(K, dtype=int)
for t in range(500):
    samples = rng.beta(alphas, betas)
    a = int(np.argmax(samples))
    r = rng.uniform() < true_p[a]
    alphas[a] += r
    betas[a] += 1 - r
    pulls[a] += 1
print(f"Thompson (true best=2): pulls per arm = {pulls}")

# DDPG/TD3/SAC — концепты
print("\nDDPG: det. actor mu(s), critic Q(s,a); explore: mu(s) + noise")
print("TD3: 2 critic, target smoothing pi'(s') + clip noise, delayed actor update")
print("SAC: stochastic actor, max E[R + alpha * H(pi)]; auto-tune alpha")
