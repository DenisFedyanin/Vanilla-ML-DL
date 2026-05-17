"""Раздел 35 — CNN. Файл 3: ResNet, ResNeXt, DenseNet, Wide ResNet.

Концепция: ResNet (He, 2015) и skip connection.
Блок: y = F(x) + x, где F — две-три свёртки. Skip-connection пропускает градиент
напрямую, что позволяет тренировать 100+ слоёв. Победил ImageNet-2015.
Bottleneck-вариант: 1x1 -> 3x3 -> 1x1 (для глубоких ResNet-50/101/152).

Концепция: ResNeXt (cardinality).
Вместо увеличения ширины или глубины — увеличивают число параллельных веток.
Каждая ветка одинаковой структуры, их выходы суммируются. Cardinality=32 типично.

Концепция: DenseNet.
Каждый слой получает на вход конкатенацию всех предыдущих feature maps.
Сильнее переиспользование признаков, меньше параметров. growth_rate=k каналов на слой.

Концепция: Wide ResNet.
Менее глубокий, но шире (больше каналов). Часто лучше при том же числе параметров.

Концепция: gradient flow через skip.
d/dx (F(x)+x) = F'(x) + I. Единичная матрица гарантирует, что градиент не зануляется.
"""
import numpy as np

rng = np.random.default_rng(0)


def relu(x):
    return np.maximum(0, x)


# === ResNet basic block: y = relu(F(x) + x) ===
print("=== ResNet basic block ===")
B, C, H, W = 1, 16, 8, 8
x = rng.standard_normal((B, C, H, W))
# F(x): имитируем две свёртки 3x3 padding=1 (форма сохраняется)
# Тут не делаем настоящую свёртку, просто шумовая трансформация той же формы
Fx = rng.standard_normal(x.shape) * 0.1
y = relu(Fx + x)
print(f"x{x.shape}, F(x){Fx.shape}, out = relu(F(x)+x){y.shape}")
print(f"норма x={np.linalg.norm(x):.2f}, norm F(x)={np.linalg.norm(Fx):.2f}, norm y={np.linalg.norm(y):.2f}")

# Bottleneck-вариант: 1x1 (C/4) -> 3x3 (C/4) -> 1x1 (C)
print("Bottleneck ResNet-50: 1x1(64) -> 3x3(64) -> 1x1(256), residual = identity (+ опц. 1x1 projection)")

# === Демонстрация gradient flow ===
print("\n=== Gradient flow через skip ===")
# Пусть градиент сверху = g. Тогда dL/dx = g * (1 + F'(x)).
# Даже если F'(x) -> 0, градиент = g не зануляется.
g_up = np.ones_like(x)
F_prime = np.zeros_like(x)   # имитируем затухание градиентов F
grad_with_skip = g_up * (1 + F_prime)
grad_without_skip = g_up * F_prime
print(f"С skip: средний |grad|={np.abs(grad_with_skip).mean():.3f}")
print(f"Без skip (если F' -> 0): средний |grad|={np.abs(grad_without_skip).mean():.3f}")

# === ResNeXt: cardinality ===
print("\n=== ResNeXt ===")
cardinality, group_c = 32, 4
total = cardinality * group_c
print(f"32 параллельные ветки по {group_c} каналов -> сумма каналов {total} (эквивалент ResNet bottleneck 128)")

# === DenseNet: concat ===
print("\n=== DenseNet block (growth_rate=12) ===")
k_g = 12
c = 24
for layer in range(4):
    new = k_g
    c_after = c + new
    print(f"  layer{layer+1}: вход={c} конкат {new} новых -> выход={c_after}")
    c = c_after

# === Wide ResNet ===
print("\nWide ResNet: depth=28, widen=10 -> те же блоки, но 10x больше каналов")
