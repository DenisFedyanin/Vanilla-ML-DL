"""Раздел 28 — Слои нейронных сетей. Файл 2: Conv1D/Conv2D и варианты.

Концепция 284: Conv1D вручную.
Скользящее окно по 1D сигналу: y[i] = sum_k x[i+k] * w[k] + b.
Применение: временные ряды, аудио, текст (символы/токены).
Ядро размера K, выход длины L - K + 1 при valid padding.

Концепция 285: Conv2D вручную.
Двумерная свёртка по картинке: y[i,j] = sum_{kh,kw,c} x[i+kh,j+kw,c]*w[kh,kw,c].
Реализуется тройным циклом по позициям и каналам. Основа CV.

Концепция 286: Padding "valid" vs "same".
"valid" — без паддинга, выход уменьшается. "same" — дополняем нулями,
чтобы выход совпал по размеру со входом (при stride=1).

Концепция 287: Strided conv.
Шаг > 1: ядро прыгает через stride пикселей. Выход уменьшается:
H_out = (H - K) // stride + 1. Альтернатива пулингу.

Концепция 288: Dilated / atrous conv.
Между элементами ядра вставляется dilation-1 промежутков.
Эффективное ядро K_eff = (K-1)*d + 1. Расширяет receptive field без роста params.
"""
import numpy as np

rng = np.random.default_rng(0)

# === 284: Conv1D ===
def conv1d(x, w, b=0.0):
    L, K = len(x), len(w)
    out = np.zeros(L - K + 1)
    for i in range(L - K + 1):
        out[i] = np.dot(x[i:i + K], w) + b
    return out

x1 = np.arange(8, dtype=float)
w1 = np.array([1.0, 0.0, -1.0])      # детектор градиента
y1 = conv1d(x1, w1)
print(f"284 Conv1D: x={x1.tolist()}, w={w1.tolist()} -> y={y1.tolist()}")

# === 285: Conv2D ===
def conv2d(x, w):
    H, W_ = x.shape
    KH, KW = w.shape
    out = np.zeros((H - KH + 1, W_ - KW + 1))
    for i in range(H - KH + 1):
        for j in range(W_ - KW + 1):
            out[i, j] = (x[i:i + KH, j:j + KW] * w).sum()
    return out

img = rng.standard_normal((5, 5))
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
y2 = conv2d(img, sobel_x)
print(f"285 Conv2D: 5x5 * 3x3 Sobel-X -> {y2.shape}, max|y|={np.abs(y2).max():.2f}")

# === 286: Padding valid vs same ===
def pad_same(x, k):
    p = (k - 1) // 2
    return np.pad(x, p)

y_valid = conv1d(x1, w1)
y_same = conv1d(pad_same(x1, 3), w1)
print(f"286 Padding: valid len={len(y_valid)}, same len={len(y_same)} (=исходная {len(x1)})")

# === 287: Strided conv ===
def conv1d_strided(x, w, s=1):
    K = len(w)
    L_out = (len(x) - K) // s + 1
    out = np.zeros(L_out)
    for i in range(L_out):
        out[i] = np.dot(x[i * s:i * s + K], w)
    return out

y_s2 = conv1d_strided(x1, w1, s=2)
print(f"287 Strided s=2: out len={len(y_s2)} (={len(x1)}-K)/2+1)")

# === 288: Dilated conv ===
def conv1d_dilated(x, w, d=1):
    K = len(w)
    K_eff = (K - 1) * d + 1
    out = np.zeros(len(x) - K_eff + 1)
    for i in range(len(out)):
        out[i] = sum(x[i + k * d] * w[k] for k in range(K))
    return out

y_d2 = conv1d_dilated(x1, w1, d=2)
print(f"288 Dilated d=2: эфф. ядро=5, out len={len(y_d2)}")
