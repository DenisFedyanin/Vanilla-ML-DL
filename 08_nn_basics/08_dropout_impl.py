"""Концепция 83: Dropout (реализация в forward/eval).

Train: маска * activations / (1-p). Eval: без маски.
"""
import numpy as np


def dropout(x, p, train=True, rng=None):
    if not train or p == 0:
        return x
    rng = rng or np.random.default_rng()
    mask = (rng.uniform(size=x.shape) > p).astype(x.dtype)
    return x * mask / (1 - p)


rng = np.random.default_rng(0)
a = np.ones(10)
print("Train mode :", dropout(a, 0.5, train=True, rng=rng))
print("Eval mode  :", dropout(a, 0.5, train=False))
