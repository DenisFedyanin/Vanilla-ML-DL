"""Концепция 117: Q-learning - вкус Reinforcement Learning.

Агент в GridWorld 5x5 учится идти к награде. Q(s,a) обновляется:
Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s', a') - Q(s,a)).
"""
import numpy as np

rng = np.random.default_rng(0)
GRID = 5
GOAL = (4, 4)
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
Q = np.zeros((GRID, GRID, 4))


def step(state, a):
    r, c = state
    dr, dc = ACTIONS[a]
    nr, nc = max(0, min(GRID - 1, r + dr)), max(0, min(GRID - 1, c + dc))
    new_state = (nr, nc)
    reward = 1.0 if new_state == GOAL else -0.01
    done = new_state == GOAL
    return new_state, reward, done


alpha, gamma, eps = 0.1, 0.9, 0.2
for ep in range(2000):
    s = (0, 0)
    for _ in range(100):
        if rng.uniform() < eps:
            a = rng.integers(4)
        else:
            a = int(Q[s].argmax())
        s2, r, done = step(s, a)
        Q[s + (a,)] += alpha * (r + gamma * Q[s2].max() - Q[s + (a,)])
        s = s2
        if done:
            break

# Покажем выученную политику
arrows = "↑↓←→"
policy = np.array([[arrows[int(Q[r, c].argmax())] for c in range(GRID)]
                   for r in range(GRID)])
policy[GOAL] = "G"
print("Выученная политика (направление действия в каждой клетке):")
for row in policy:
    print(" ".join(row))
