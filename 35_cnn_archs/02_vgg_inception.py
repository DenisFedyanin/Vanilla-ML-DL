"""Раздел 35 — CNN. Файл 2: VGG, NiN, Inception/GoogLeNet.

Концепция: VGG (Simonyan, 2014).
Идея: вместо больших ядер (7x7) стекать маленькие 3x3.
Два 3x3 имеют рецептивное поле 5x5, но меньше параметров и больше нелинейностей.
VGG-16: 13 conv + 3 FC. Простая и однородная архитектура.

Концепция: NiN (Network in Network).
Между обычными свёртками вставляют 1x1 conv ("micro-MLP" по каналам).
1x1 conv = линейная комбинация каналов; уменьшает число каналов (bottleneck).
Заменяет FC слой через GlobalAveragePooling.

Концепция: Inception module (GoogLeNet, 2014).
Параллельные ветки 1x1, 3x3, 5x5, MaxPool — их выходы конкатенируются по каналам.
Чтобы снизить стоимость, перед 3x3/5x5 ставят bottleneck 1x1.
Многомасштабные признаки в одном блоке.

Концепция: вспомогательные классификаторы (auxiliary heads).
GoogLeNet имел дополнительные классификаторы из промежуточных слоёв.
Их градиенты помогали обучать глубокую сеть и регуляризовали её.
"""
import numpy as np


def conv_shape(s, k, p=0, st=1):
    return (s + 2 * p - k) // st + 1


# === VGG: два 3x3 vs одно 5x5 ===
print("=== VGG: 3x3 стек ===")
C = 64
p_5x5 = 5 * 5 * C * C
p_3x3x2 = 2 * (3 * 3 * C * C)
print(f"Параметры 5x5 conv (C={C}): {p_5x5}; два 3x3: {p_3x3x2} (меньше на {p_5x5-p_3x3x2})")
print("Рецептивное поле одинаковое = 5x5")

# === VGG flow: 224 -> ... ===
H = 224
for stage, (n_conv, out_c) in enumerate([(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)], 1):
    # n_conv свёрток 3x3 padding=1 не меняют размер
    H = conv_shape(H, k=2, st=2)  # после блока — MaxPool 2x2 stride=2
    print(f"VGG stage{stage} ({n_conv} conv 3x3->{out_c}) + pool -> H={H}")

# === NiN: 1x1 conv ===
print("\n=== NiN 1x1 ===")
in_c, out_c = 256, 64
params = 1 * 1 * in_c * out_c + out_c
print(f"1x1 conv {in_c}->{out_c}: params={params} — bottleneck по каналам")

# === Inception module: параллельные ветки ===
print("\n=== Inception (выход 28x28x256, четыре ветки) ===")
branches = {
    "1x1": 64,
    "3x3 (после 1x1 bottleneck=96)": 128,
    "5x5 (после 1x1 bottleneck=16)": 32,
    "MaxPool + 1x1": 32,
}
total = sum(branches.values())
for k, v in branches.items():
    print(f"  ветка {k}: out_ch={v}")
print(f"  concat по каналам: total={total}")

# === GoogLeNet auxiliary heads ===
print("\nGoogLeNet: 2 вспомогательные головы из промежуточных слоёв,")
print("loss = main + 0.3*aux1 + 0.3*aux2 (только при обучении)")

# === Manual shape demo ===
rng = np.random.default_rng(0)
x = rng.standard_normal((1, 192, 28, 28))
# 4 ветки концептуально
out = np.zeros((1, total, 28, 28))
print(f"\nManual: x{x.shape} -> Inception -> y{out.shape}")
