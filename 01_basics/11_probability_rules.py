"""Концепция 11: Сложение и умножение вероятностей.

Демонстрация на симуляции бросков двух кубиков.
"""
import numpy as np

rng = np.random.default_rng(1)
N = 200_000
d1 = rng.integers(1, 7, N)
d2 = rng.integers(1, 7, N)

A = d1 == 6                         # 1й = 6
B = d2 == 6                         # 2й = 6

p_A = A.mean()
p_B = B.mean()
p_AandB = (A & B).mean()
p_AorB = (A | B).mean()

print(f"P(A)            = {p_A:.4f}  (теор. ~0.1667)")
print(f"P(B)            = {p_B:.4f}")
print(f"P(A и B) ~ P(A)*P(B) = {p_AandB:.4f}  (теор. ~0.0278)")
print(f"P(A или B) = P(A)+P(B)-P(AиB) = {p_AorB:.4f}  (теор. ~0.3056)")
