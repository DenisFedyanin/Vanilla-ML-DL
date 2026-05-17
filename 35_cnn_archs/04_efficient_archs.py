"""Раздел 35 — CNN. Файл 4: эффективные архитектуры для мобильных.

Концепция: MobileNet — depthwise separable conv.
Заменяет обычную свёртку (k*k*Cin*Cout параметров) на:
  depthwise: k*k*Cin (каждый канал свёртывается отдельно) +
  pointwise: 1*1*Cin*Cout.
Параметров в ~k^2 раз меньше при близком качестве.

Концепция: MobileNetV2 — inverted residual + linear bottleneck.
expand 1x1 (увеличить каналы) -> depthwise 3x3 -> project 1x1 (уменьшить, без ReLU).
Skip-connection только когда вход и выход одной формы.

Концепция: ShuffleNet — group conv + channel shuffle.
Свёртка по группам экономит вычисления; затем channel shuffle перемешивает
каналы между группами, чтобы информация смешивалась.

Концепция: SqueezeNet — fire module.
Squeeze 1x1 (мало каналов) -> Expand параллельные 1x1 и 3x3 -> concat.
Дает AlexNet-уровень при 50x меньше параметров.

Концепция: EfficientNet — compound scaling.
Масштабируют ширину w=α^φ, глубину d=β^φ, разрешение r=γ^φ одновременно,
при α*β^2*γ^2 ≈ 2. Один коэффициент φ управляет «бюджетом».
"""
import numpy as np

# === MobileNet depthwise separable ===
print("=== Depthwise separable vs standard conv ===")
Cin, Cout, k = 64, 128, 3
standard = k * k * Cin * Cout
depthwise = k * k * Cin
pointwise = 1 * 1 * Cin * Cout
sep = depthwise + pointwise
print(f"Standard conv {k}x{k} {Cin}->{Cout}: params={standard}")
print(f"Depthwise: {depthwise}, Pointwise(1x1): {pointwise}, total separable: {sep}")
print(f"Экономия: {standard/sep:.1f}x")

# Демонстрация форм
H = 32
in_t = np.zeros((1, Cin, H, H))
dw = np.zeros((1, Cin, H, H))     # depthwise сохраняет число каналов
pw = np.zeros((1, Cout, H, H))    # pointwise меняет каналы
print(f"shapes: in{in_t.shape} -> dw{dw.shape} -> pw{pw.shape}")

# === MobileNetV2 inverted residual ===
print("\n=== MobileNetV2 inverted residual (expand=6) ===")
in_c, out_c = 24, 24
exp = 6 * in_c
print(f"  1x1 expand: {in_c} -> {exp} (+ReLU6)")
print(f"  3x3 depthwise stride=1: {exp} -> {exp} (+ReLU6)")
print(f"  1x1 project (LINEAR): {exp} -> {out_c}")
print("  residual если in_c==out_c и stride=1")

# === ShuffleNet ===
print("\n=== ShuffleNet ===")
C, G = 24, 3
# channel shuffle: reshape (C/G, G) -> transpose -> flatten
x = np.arange(C)
shuffled = x.reshape(G, C // G).T.reshape(-1)
print(f"channels before: {x.tolist()}")
print(f"channels after shuffle (groups=3): {shuffled.tolist()}")

# === SqueezeNet fire module ===
print("\n=== SqueezeNet fire (squeeze=16, expand=64+64) ===")
sq = 16
e1, e3 = 64, 64
print(f"  squeeze 1x1: in->{sq}")
print(f"  expand 1x1: {sq}->{e1}; expand 3x3: {sq}->{e3}")
print(f"  concat по каналам: {e1+e3}")

# === EfficientNet compound scaling ===
print("\n=== EfficientNet compound scaling ===")
alpha, beta, gamma = 1.2, 1.1, 1.15
for phi in [0, 1, 2, 3]:
    d = beta ** phi
    w = alpha ** phi
    r = gamma ** phi
    flops = (d * w * w * r * r)
    print(f"phi={phi}: depth*={d:.2f}, width*={w:.2f}, res*={r:.2f}, flops*≈{flops:.2f}")
print("Формула: w=α^φ, d=β^φ, r=γ^φ; α*β^2*γ^2 ≈ 2 -> flops удваиваются с +1 к φ")
