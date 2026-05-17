"""Раздел 35 — CNN архитектуры. Файл 1: LeNet, AlexNet.

Концепция: LeNet-5 (LeCun, 1998).
Классическая архитектура для рукописных цифр: Conv -> Pool -> Conv -> Pool -> FC -> FC.
Использовала tanh/sigmoid и AveragePool. Обучалась на MNIST.
Показала, что свёртки + субдискретизация работают для изображений.

Концепция: AlexNet (Krizhevsky, 2012).
Прорыв ImageNet 2012: ReLU вместо tanh, Dropout в FC, обучение на двух GPU.
Стек 5 свёрточных + 3 FC. Использовал LRN (Local Response Normalization).
ReLU ускорил обучение в разы; Dropout победил переобучение.

Концепция: расчёт размеров после Conv.
out = (in + 2*pad - kernel) // stride + 1. Pool делит размер.
Параметры Conv: kernel*kernel*in_ch*out_ch + out_ch (bias).
"""
import numpy as np


def conv_shape(in_size, k, s=1, p=0):
    return (in_size + 2 * p - k) // s + 1


def pool_shape(in_size, k, s=None):
    s = s or k
    return (in_size - k) // s + 1


# === LeNet-5 на 28x28 (MNIST) ===
print("=== LeNet-5 (вход 1x28x28) ===")
H = 28
H = conv_shape(H, k=5, p=2)     # Conv1: 6 фильтров, padding 2 -> 28
print(f"Conv1 5x5 (6 фильтров, pad=2): 28 -> {H}, params={5*5*1*6 + 6}")
H = pool_shape(H, k=2)           # Pool1 -> 14
print(f"AvgPool 2x2: -> {H}")
H = conv_shape(H, k=5)           # Conv2: 16 фильтров -> 10
print(f"Conv2 5x5 (16 фильтров): -> {H}, params={5*5*6*16 + 16}")
H = pool_shape(H, k=2)           # Pool2 -> 5
print(f"AvgPool 2x2: -> {H}")
flat = H * H * 16
print(f"Flatten -> {flat}; FC(120) params={flat*120 + 120}; FC(84); FC(10)")

# === AlexNet (концептуально на 227x227, тут просто показать формулы) ===
print("\n=== AlexNet (вход 3x227x227) ===")
H = 227
H = conv_shape(H, k=11, s=4)    # Conv1 96 фильтров, stride 4
print(f"Conv1 11x11 s=4 (96): -> {H}, ReLU, LRN")
H = pool_shape(H, k=3, s=2)
print(f"MaxPool 3x3 s=2: -> {H}")
H = conv_shape(H, k=5, p=2)
print(f"Conv2 5x5 pad=2 (256): -> {H}")
H = pool_shape(H, k=3, s=2)
print(f"MaxPool: -> {H}")
print("Conv3-5 (3x3, padding=1) сохраняют размер, потом MaxPool, потом FC(4096)->FC(4096)->FC(1000)")
print("Dropout 0.5 в двух FC; ReLU после каждого слоя")

# === Демонстрация: manual forward на dummy 28x28 (только формы) ===
rng = np.random.default_rng(0)
x = rng.standard_normal((1, 1, 28, 28))   # B,C,H,W
W1 = rng.standard_normal((6, 1, 5, 5)) * 0.1
# тут не реализуем настоящую свёртку, просто демонстрируем форму выхода
out_h = conv_shape(28, 5, p=2)
y = np.zeros((1, 6, out_h, out_h))
print(f"\nManual shapes: x{x.shape} -> Conv1 -> y{y.shape}")
