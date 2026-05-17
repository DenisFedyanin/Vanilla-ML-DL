"""Концепция 91: Max и Average Pooling.

После свёртки часто делают pooling 2x2: берут максимум (или среднее)
по окну. Это уменьшает размер и даёт инвариантность к небольшим сдвигам.
"""
import numpy as np


def pool(img, size=2, mode="max"):
    H, W = img.shape
    out = np.zeros((H // size, W // size))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            window = img[i * size:(i + 1) * size, j * size:(j + 1) * size]
            out[i, j] = window.max() if mode == "max" else window.mean()
    return out


img = np.array([[1, 3, 2, 4], [5, 6, 7, 8],
                [9, 10, 11, 12], [13, 14, 15, 16.0]])
print("Max-pool 2x2:\n", pool(img, 2, "max"))
print("Avg-pool 2x2:\n", pool(img, 2, "avg"))
