"""Раздел 28 — Слои нейронных сетей. Файл 3: спец. свёртки.

Концепция 289: ConvTranspose / Upsampling.
"Обратная" свёртка: между пикселями входа вставляются нули,
потом обычная свёртка. Используется для повышения разрешения (U-Net, GAN-генератор).

Концепция 290: Depthwise convolution.
Каждый канал свёртывается СВОИМ ядром, каналы не смешиваются.
Параметров в C раз меньше, чем у обычной свёртки. База MobileNet.

Концепция 291: Pointwise (1x1) conv.
Свёртка 1x1 по пространству смешивает каналы. Эквивалентна Dense по C.
Используется для изменения числа каналов и микширования.

Концепция 292: Separable convolution.
Depthwise + Pointwise = почти полноценная свёртка, но в разы меньше параметров.
MobileNet, Xception. Главный приём для лёгких CV-моделей.
"""
import numpy as np

rng = np.random.default_rng(0)

# === 289: ConvTranspose (upsample) ===
def upsample_conv(x, w, factor=2):
    # вставим (factor-1) нулей между каждой парой элементов
    L = len(x)
    up = np.zeros(L * factor)
    up[::factor] = x
    K = len(w)
    out = np.zeros(len(up) - K + 1)
    for i in range(len(out)):
        out[i] = np.dot(up[i:i + K], w)
    return out

x = np.array([1.0, 2.0, 3.0])
w = np.array([0.5, 1.0, 0.5])
y = upsample_conv(x, w, factor=2)
print(f"289 ConvTranspose: вход len={len(x)} -> выход len={len(y)}")

# === 290: Depthwise conv ===
def depthwise_conv2d(x, kernels):
    # x: (C, H, W), kernels: (C, KH, KW) — по ядру на канал
    C, H, W_ = x.shape
    _, KH, KW = kernels.shape
    out = np.zeros((C, H - KH + 1, W_ - KW + 1))
    for c in range(C):
        for i in range(H - KH + 1):
            for j in range(W_ - KW + 1):
                out[c, i, j] = (x[c, i:i + KH, j:j + KW] * kernels[c]).sum()
    return out

C, H, W = 3, 5, 5
img = rng.standard_normal((C, H, W))
ker_dw = rng.standard_normal((C, 3, 3)) * 0.1
y_dw = depthwise_conv2d(img, ker_dw)
print(f"290 Depthwise: in{img.shape} -> out{y_dw.shape}, params=C*KH*KW={C * 3 * 3}")

# === 291: Pointwise (1x1) conv ===
def pointwise_conv2d(x, w_pw):
    # x: (C_in, H, W), w_pw: (C_out, C_in)
    return np.einsum("oi,ihw->ohw", w_pw, x)

C_out = 4
w_pw = rng.standard_normal((C_out, C)) * 0.1
y_pw = pointwise_conv2d(y_dw, w_pw)
print(f"291 Pointwise: {y_dw.shape} -> {y_pw.shape}, params=C_out*C_in={C_out * C}")

# === 292: Separable conv = DW + PW ===
def separable_conv2d(x, dw_kernels, pw):
    return pointwise_conv2d(depthwise_conv2d(x, dw_kernels), pw)

y_sep = separable_conv2d(img, ker_dw, w_pw)
params_sep = C * 3 * 3 + C_out * C
params_full = C_out * C * 3 * 3
print(f"292 Separable: out{y_sep.shape}; params {params_sep} vs full {params_full} (экономия ~{params_full / params_sep:.1f}x)")
