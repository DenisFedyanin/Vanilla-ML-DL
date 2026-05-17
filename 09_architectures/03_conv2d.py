"""Концепция 90: 2D-свёртка - вручную.

То же самое окошко, но 2D - идёт по строкам и столбцам изображения.
"""
import numpy as np


def conv2d(img, k):
    H, W = img.shape; kh, kw = k.shape
    out = np.zeros((H - kh + 1, W - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(img[i:i + kh, j:j + kw] * k)
    return out


img = np.array([
    [1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0],
    [0, 0, 0, 1, 1],
    [0, 0, 0, 1, 1.0],
])
edge_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])  # вертикальный край
print("Свёртка с детектором вертикальных краёв:\n", conv2d(img, edge_x))
