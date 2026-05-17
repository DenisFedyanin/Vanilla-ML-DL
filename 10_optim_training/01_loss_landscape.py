"""Концепция 98: Loss landscape.

Loss как функция параметров - 'рельеф', по которому скатывается оптимизатор.
Бывают плоские плато, овраги, локальные минимумы, седловые точки.
"""
import numpy as np


def loss(w1, w2):
    return (w1 - 1) ** 2 + 5 * (w2 + 2) ** 2 + 0.1 * np.sin(5 * w1) * np.cos(5 * w2)


w1 = np.linspace(-3, 3, 5)
w2 = np.linspace(-5, 1, 5)
print("Loss значения по сетке параметров:")
for v1 in w1:
    for v2 in w2:
        print(f"  w1={v1:.1f} w2={v2:.1f} loss={loss(v1, v2):.2f}")
print("\nМинимум около (1, -2).")
