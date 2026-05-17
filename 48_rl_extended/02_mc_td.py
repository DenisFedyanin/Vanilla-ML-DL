"""Раздел 48 — RL расширенный.

Файл 2: Monte Carlo и TD-обучение.

Концепция: Monte Carlo prediction.
Учим V(s) как среднее по возврату G_t = R_{t+1} + gamma R_{t+2} + ... в эпизодах.
Every-visit: усредняем по всем посещениям; first-visit: только по первому.
Несмещённое, но высокая дисперсия и нужны эпизоды.

Концепция: TD(0).
V(s) <- V(s) + alpha [R + gamma V(s') - V(s)]. Использует bootstrap (V(s')).
Низкая дисперсия, online, без эпизодов. Смещено в начале обучения.

Концепция: n-step TD.
G_t^{(n)} = R_{t+1} + ... + gamma^{n-1} R_{t+n} + gamma^n V(s_{t+n}).
Промежуточное между MC и TD: n=1 — TD(0), n=inf — MC.

Концепция: TD(lambda) и eligibility traces.
Геометрическая смесь n-step возвратов с весами (1-lambda)*lambda^{n-1}.
Эквивалент: backup через накапливаемые traces e(s) на каждое посещение.

Концепция: Bias-variance.
MC: высокая var, низкий bias. TD(0): низкая var, есть bias. TD(lambda) с
правильным lambda — компромисс.
"""
import numpy as np

rng = np.random.default_rng(0)
# Простой коридор: 5 состояний 0..4, цель 4 (R=+1), штраф 0 (R=-1), внутри R=0
n_states = 5
gamma = 0.9


def step_corridor(s, a):
    # a: 0=left, 1=right
    sp = s + (1 if a == 1 else -1)
    sp = max(0, min(n_states - 1, sp))
    if sp == 0:
        return sp, -1.0, True
    if sp == n_states - 1:
        return sp, 1.0, True
    return sp, 0.0, False


def gen_episode(pi="random", max_len=50):
    s = 2
    traj = []
    for _ in range(max_len):
        a = rng.integers(0, 2) if pi == "random" else 1
        sp, r, done = step_corridor(s, a)
        traj.append((s, a, r, sp))
        s = sp
        if done:
            break
    return traj


# Monte Carlo every-visit
V_mc = np.zeros(n_states)
counts = np.zeros(n_states)
for _ in range(2000):
    ep = gen_episode("random")
    G = 0
    for (s, a, r, sp) in reversed(ep):
        G = r + gamma * G
        counts[s] += 1
        V_mc[s] += (G - V_mc[s]) / counts[s]
print("Monte Carlo V:", V_mc.round(3))

# TD(0)
V_td = np.zeros(n_states)
alpha = 0.1
for _ in range(2000):
    ep = gen_episode("random")
    for (s, a, r, sp) in ep:
        V_td[s] += alpha * (r + gamma * V_td[sp] - V_td[s])
print("TD(0)       V:", V_td.round(3))

# n-step TD (n=3)
V_n = np.zeros(n_states)
n = 3
for _ in range(2000):
    ep = gen_episode("random")
    T_len = len(ep)
    for t in range(T_len):
        G = 0
        for k in range(n):
            if t + k >= T_len:
                break
            G += (gamma ** k) * ep[t + k][2]
        if t + n < T_len:
            G += (gamma ** n) * V_n[ep[t + n][0]]
        s = ep[t][0]
        V_n[s] += alpha * (G - V_n[s])
print(f"n-step TD (n={n}) V:", V_n.round(3))

# TD(lambda) с накопительными traces
V_lam = np.zeros(n_states)
lam = 0.5
for _ in range(2000):
    e = np.zeros(n_states)
    ep = gen_episode("random")
    for (s, a, r, sp) in ep:
        delta = r + gamma * V_lam[sp] - V_lam[s]
        e[s] += 1
        V_lam += alpha * delta * e
        e *= gamma * lam
print(f"TD(λ={lam})     V:", V_lam.round(3))
print("\nИстинная V: должна расти от -1 к +1, в центре ~0 (для random policy)")
