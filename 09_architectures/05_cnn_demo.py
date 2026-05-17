"""Концепция 92: CNN - концепт (свёртка + ReLU + pooling).

Маленький forward pass: вход 8x8 -> conv -> relu -> max-pool.
"""
import numpy as np


def conv2d(img, k):
    H, W = img.shape; kh, kw = k.shape
    out = np.zeros((H - kh + 1, W - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(img[i:i + kh, j:j + kw] * k)
    return out


def maxpool(x, s=2):
    H, W = x.shape
    return np.array([[x[i * s:(i + 1) * s, j * s:(j + 1) * s].max()
                      for j in range(W // s)] for i in range(H // s)])


rng = np.random.default_rng(0)
img = rng.normal(size=(8, 8))
k = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1.0]])

c = conv2d(img, k)
a = np.maximum(0, c)
p = maxpool(a, 2)
print("Shape вход:", img.shape)
print("Shape после conv:", c.shape)
print("Shape после relu+pool:", p.shape)
