"""Раздел 42 — Computer Vision.

Файл 2: свойства свёртки.

Концепция: Линейность свёртки.
conv(a*x + b*y, k) = a*conv(x,k) + b*conv(y,k). Свёртка — линейный оператор.

Концепция: Shift-equivariance.
Если сдвинуть вход, выход свёртки сдвинется так же. Базовое свойство CNN —
позволяет распознавать объекты независимо от их положения.

Концепция: Режимы padding.
'valid' — без паддинга, выход меньше. 'same' — паддинг такой, чтобы выход
совпал по размеру с входом. 'full' — максимальный паддинг, выход больше.

Концепция: Receptive field (рецептивное поле).
Какую область входа "видит" один выходной нейрон. Для стека свёрток:
RF_{n} = RF_{n-1} + (k_n - 1) * П(stride_{i<n}).

Концепция: Stride и dilation.
stride — шаг по выходу. dilation — пропуски между элементами ядра.
Dilated conv увеличивает RF без роста числа параметров.
"""
import numpy as np


def conv1d(x, k, mode="valid"):
    """Свёртка 1D (корреляция: без переворота ядра — как в нейросетях)."""
    n, m = len(x), len(k)
    if mode == "valid":
        out = np.zeros(n - m + 1)
        for i in range(len(out)):
            out[i] = (x[i:i + m] * k).sum()
        return out
    if mode == "same":
        p = (m - 1) // 2
        xp = np.pad(x, p)
        return conv1d(xp, k, "valid")[:n]
    if mode == "full":
        p = m - 1
        xp = np.pad(x, p)
        return conv1d(xp, k, "valid")
    raise ValueError(mode)


rng = np.random.default_rng(0)
x = rng.normal(size=10)
y = rng.normal(size=10)
k = np.array([1.0, -1.0, 0.5])

# Линейность
a, b = 2.0, -3.0
lhs = conv1d(a * x + b * y, k)
rhs = a * conv1d(x, k) + b * conv1d(y, k)
print("Линейность max|разница|:", np.abs(lhs - rhs).max())

# Shift-equivariance: conv(shift(x)) == shift(conv(x))
x_shift = np.roll(x, 2)
out1 = conv1d(x_shift, k)
out2 = np.roll(conv1d(x, k), 2)
# Сравним внутреннюю часть (без артефактов края roll)
print("Shift-equivariance max|разница| (внутри):", np.abs(out1[2:] - out2[2:]).max().round(3))

# Padding-режимы
print("valid длина:", len(conv1d(x, k, "valid")), "(=10-3+1=8)")
print("same  длина:", len(conv1d(x, k, "same")), "(=10)")
print("full  длина:", len(conv1d(x, k, "full")), "(=10+3-1=12)")

# Receptive field для стека: k=3, stride=1, 3 слоя -> 1 + 3*(3-1) = 7
def rf_stack(kernels, strides):
    rf, jump = 1, 1
    for k_, s_ in zip(kernels, strides):
        rf += (k_ - 1) * jump
        jump *= s_
    return rf


print("RF [3,3,3] stride 1:", rf_stack([3, 3, 3], [1, 1, 1]))
print("RF [3,3,3] stride 2:", rf_stack([3, 3, 3], [2, 2, 2]))
print("RF [3,3] dilated (эквивалент k=5,5):", rf_stack([5, 5], [1, 1]))

# Dilated conv демо: ядро [a,b,c] с dilation=2 -> вес на позициях 0,2,4
xd = np.arange(8.0)
kd = np.array([1.0, 1.0, 1.0])
dil = 2
out_dil = np.zeros(len(xd) - (len(kd) - 1) * dil)
for i in range(len(out_dil)):
    out_dil[i] = sum(kd[j] * xd[i + j * dil] for j in range(len(kd)))
print("Dilated conv output:", out_dil)
