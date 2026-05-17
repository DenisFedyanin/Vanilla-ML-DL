"""Концепция 112: KL-дивергенция и связь с cross-entropy.

KL(P || Q) = sum p * log(p/q) - 'насколько Q плохо приближает P'.
H(P, Q) = H(P) + KL(P || Q).  Минимизируя CE, мы минимизируем KL.
"""
import numpy as np

p = np.array([0.7, 0.2, 0.1])
q1 = np.array([0.6, 0.3, 0.1])      # близко к p
q2 = np.array([0.1, 0.1, 0.8])      # далеко от p


def kl(p, q): return np.sum(p * np.log(p / q))


print(f"KL(p || q_близкий) = {kl(p, q1):.3f}")
print(f"KL(p || q_далёкий) = {kl(p, q2):.3f}")
