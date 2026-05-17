"""Концепция 4: Косинусное сходство.

Измеряет угол между векторами: 1 = совпадают, 0 = перпендикулярны, -1 = противоположны.
Используется в рекомендациях и эмбеддингах.
"""
import numpy as np


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


users = {
    "Аня": np.array([5, 4, 0, 1]),
    "Боря": np.array([4, 5, 0, 1]),
    "Вика": np.array([0, 1, 5, 4]),
}

for u1 in users:
    for u2 in users:
        if u1 < u2:
            print(f"cos({u1},{u2}) = {cosine(users[u1], users[u2]):.3f}")
