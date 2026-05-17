"""Раздел 42 — Computer Vision.

Файл 3: ускорение свёртки.

Концепция: im2col.
Превращаем все патчи KxK из (H,W) в строки матрицы (N, K*K). После этого
свёртка = матричное умножение matrix @ kernel.flatten(). Так делают GPU/BLAS.

Концепция: Свёртка через matmul.
Когда im2col готов, conv = (patches @ kernels). Один батч изображений ->
(N, C*K*K) @ (C*K*K, F) — высоко-оптимизировано через GEMM.

Концепция: Свёртка через FFT.
В частотной области свёртка = поэлементное произведение. Для больших ядер
быстрее: O(N log N) против O(N*K^2). На малых K im2col эффективнее.

Концепция: Сепарабельные фильтры.
2D-ядро G(x,y) разлагается G = g(x) * g(y)^T (внешнее произведение).
Тогда conv2d = conv_x(conv_y(.)). Стоимость K^2 -> 2K на пиксель.
Gaussian — классический сепарабельный пример.
"""
import numpy as np


def im2col(img, k):
    """Из (H,W) делает (n_patches, k*k)."""
    H, W = img.shape
    out_h, out_w = H - k + 1, W - k + 1
    cols = np.zeros((out_h * out_w, k * k))
    idx = 0
    for i in range(out_h):
        for j in range(out_w):
            cols[idx] = img[i:i + k, j:j + k].ravel()
            idx += 1
    return cols, (out_h, out_w)


rng = np.random.default_rng(0)
img = rng.normal(size=(6, 6))
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # sharpen

cols, shape = im2col(img, 3)
print("im2col: img (6,6) -> patches", cols.shape)

# Свёртка через matmul
out_mm = (cols @ kernel.ravel()).reshape(shape)
print("Свёртка через matmul форма:", out_mm.shape)

# Свёртка обычным циклом (для сверки)
out_ref = np.zeros(shape)
for i in range(shape[0]):
    for j in range(shape[1]):
        out_ref[i, j] = (img[i:i + 3, j:j + 3] * kernel).sum()
print("matmul vs ref max|diff|:", np.abs(out_mm - out_ref).max().round(6))

# FFT-свёртка 1D
x = rng.normal(size=64)
k = np.array([0.25, 0.5, 0.25])
n = len(x) + len(k) - 1
X = np.fft.fft(x, n)
K = np.fft.fft(k, n)
y_fft = np.real(np.fft.ifft(X * K))
# Эквивалент 'full'
y_ref = np.convolve(x, k, mode="full")
print("FFT-свёртка max|diff|:", np.abs(y_fft - y_ref).max().round(8))

# Сепарабельный Gaussian: G(x,y) = g(x) * g(y)^T
sigma = 1.0
xs = np.arange(-2, 3)
g1d = np.exp(-xs ** 2 / (2 * sigma ** 2))
g1d /= g1d.sum()
G2d = np.outer(g1d, g1d)
print("Gaussian 1D:", g1d.round(3))
print("Gaussian 2D ранг (rank):", np.linalg.matrix_rank(G2d.round(8)),
      "(1 = сепарабельный)")

# Сепарабельная свёртка: сначала по строкам, потом по столбцам
img2 = rng.normal(size=(20, 20))
# вдоль строк
tmp = np.zeros((20, 16))
for i in range(20):
    tmp[i] = np.convolve(img2[i], g1d, mode="valid")
# вдоль столбцов
out_sep = np.zeros((16, 16))
for j in range(16):
    out_sep[:, j] = np.convolve(tmp[:, j], g1d, mode="valid")
# Прямая 2D-свёртка (для сверки)
out_full = np.zeros((16, 16))
for i in range(16):
    for j in range(16):
        out_full[i, j] = (img2[i:i + 5, j:j + 5] * G2d).sum()
print("Сепарабельная vs 2D max|diff|:", np.abs(out_sep - out_full).max().round(8))
