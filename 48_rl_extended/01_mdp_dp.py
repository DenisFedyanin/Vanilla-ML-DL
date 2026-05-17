"""Раздел 48 — Reinforcement Learning расширенный.

Файл 1: MDP и Dynamic Programming.

Концепция: MDP (Markov Decision Process).
Кортеж (S, A, R, P, gamma): состояния, действия, награды, переходы
P(s'|s, a), дисконт. Марковское свойство: будущее зависит только от текущего s.

Концепция: Bellman expectation.
V^pi(s) = E_pi [R + gamma V^pi(s')] = sum_a pi(a|s) sum_{s'} P(s'|s,a)[R(s,a,s') + gamma V^pi(s')].

Концепция: Bellman optimality.
V*(s) = max_a sum_{s'} P(s'|s,a)[R + gamma V*(s')].
Аналогично Q*(s, a) = sum_{s'} P[R + gamma max_{a'} Q*(s', a')].

Концепция: Value iteration.
Многократно применяем правый член Bellman optimality как обновление V.
Сходится к V* за счёт contraction mapping (gamma<1).

Концепция: Policy iteration.
Чередуем policy evaluation (V^pi) и greedy improvement (pi = argmax_a Q).
Сходится за конечное число итераций (детерминированный мир).
"""
import numpy as np

# GridWorld 3x3, цель — клетка (2, 2), штраф клетка (1, 1)
H, W = 3, 3
n_states = H * W
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # U, D, L, R
gamma = 0.9


def to_idx(i, j):
    return i * W + j


# Награды
R = np.full((n_states, 4), -0.04)
goal = to_idx(2, 2)
trap = to_idx(1, 1)
# Переходы детерминированные
def step(s, a):
    i, j = s // W, s % W
    di, dj = actions[a]
    ni, nj = i + di, j + dj
    if not (0 <= ni < H and 0 <= nj < W):
        ni, nj = i, j
    return to_idx(ni, nj)


# Терминальные состояния
terminal = {goal: 1.0, trap: -1.0}

# Value iteration
V = np.zeros(n_states)
for it in range(200):
    V_new = V.copy()
    for s in range(n_states):
        if s in terminal:
            V_new[s] = terminal[s]
            continue
        qs = []
        for a in range(4):
            sp = step(s, a)
            r = terminal.get(sp, -0.04)
            qs.append(r + gamma * V[sp])
        V_new[s] = max(qs)
    if np.abs(V_new - V).max() < 1e-6:
        break
    V = V_new
print(f"Value iteration сошлась за {it} итераций")
print("V (3x3):")
print(V.reshape(H, W).round(2))

# Извлечь жадную политику
pi = np.zeros(n_states, dtype=int)
for s in range(n_states):
    if s in terminal:
        pi[s] = -1
        continue
    qs = []
    for a in range(4):
        sp = step(s, a)
        r = terminal.get(sp, -0.04)
        qs.append(r + gamma * V[sp])
    pi[s] = int(np.argmax(qs))
arrows = ["U", "D", "L", "R"]
print("Policy (3x3):")
for i in range(H):
    print([arrows[pi[to_idx(i, j)]] if pi[to_idx(i, j)] >= 0 else "*"
           for j in range(W)])

# Policy iteration
pi2 = np.zeros(n_states, dtype=int)
for it_pi in range(50):
    # Policy evaluation
    V2 = np.zeros(n_states)
    for _ in range(200):
        Vn = V2.copy()
        for s in range(n_states):
            if s in terminal:
                Vn[s] = terminal[s]
                continue
            a = pi2[s]
            sp = step(s, a)
            r = terminal.get(sp, -0.04)
            Vn[s] = r + gamma * V2[sp]
        if np.abs(Vn - V2).max() < 1e-6:
            break
        V2 = Vn
    # Improvement
    changed = False
    for s in range(n_states):
        if s in terminal:
            continue
        qs = []
        for a in range(4):
            sp = step(s, a)
            r = terminal.get(sp, -0.04)
            qs.append(r + gamma * V2[sp])
        new_a = int(np.argmax(qs))
        if new_a != pi2[s]:
            pi2[s] = new_a
            changed = True
    if not changed:
        break
print(f"Policy iteration сошлась за {it_pi} внешних шагов")
print("V (policy iter):")
print(V2.reshape(H, W).round(2))
