"""Раздел 45 — Сегментация.

Файл 1: семантическая сегментация (FCN, U-Net, SegNet, DeepLab).

Концепция: Semantic segmentation.
Каждому пикселю присваиваем класс (без различения instance'ов). Выход —
карта тех же H, W, что и вход, с C каналами вероятностей.

Концепция: FCN (Fully Convolutional Network).
Заменяем FC-слои на 1x1 convs и используем transposed convs для upsample.
Получаем dense выходы. Skip-connections с промежуточных уровней улучшают
детали (FCN-32s/16s/8s).

Концепция: U-Net.
Encoder (downsample через pooling/strided conv) + decoder (upsample) +
skip-connections, конкатенирующие энкодерные фичи с декодером того же
разрешения. Стандарт в медицинской сегментации.

Концепция: SegNet.
Симметричный encoder-decoder. Вместо передачи фичей по skip передаём
индексы pooling — decoder делает unpooling в те же позиции.

Концепция: DeepLab + ASPP.
Atrous (dilated) convs увеличивают receptive field без снижения разрешения.
ASPP = Atrous Spatial Pyramid Pooling: параллельные ветки с разной dilation.
"""
import numpy as np


# Имитация encoder-decoder с симуляцией shape
def downsample(shape):
    return (shape[0], shape[1] // 2, shape[2] // 2)


def upsample(shape):
    return (shape[0], shape[1] * 2, shape[2] * 2)


# U-Net shape simulation
input_shape = (3, 64, 64)  # C, H, W
print("U-Net encoder:")
e1 = (32, 64, 64)
e2 = downsample((32, 64, 64))  # 32x32
e2 = (64,) + e2[1:]
e3 = (128,) + downsample(e2)[1:]
e4 = (256,) + downsample(e3)[1:]
bn = (512,) + downsample(e4)[1:]
for i, s in enumerate([input_shape, e1, e2, e3, e4, bn]):
    print(f"  уровень {i}: {s}")

print("U-Net decoder (с конкатенацией skip):")
d4 = (256,) + upsample(bn)[1:]
d4_cat = (d4[0] + e4[0],) + d4[1:]  # skip
d3 = (128,) + upsample(d4_cat)[1:]
d3_cat = (d3[0] + e3[0],) + d3[1:]
d2 = (64,) + upsample(d3_cat)[1:]
d2_cat = (d2[0] + e2[0],) + d2[1:]
d1 = (32,) + upsample(d2_cat)[1:]
d1_cat = (d1[0] + e1[0],) + d1[1:]
out = (5, 64, 64)  # 5 классов
for i, s in enumerate([bn, d4_cat, d3_cat, d2_cat, d1_cat, out]):
    print(f"  декодер шаг {i}: {s}")

# FCN: заменяем FC -> 1x1 conv. Симуляция выходов FCN-32s / 16s / 8s
print("\nFCN: FC -> 1x1 conv -> upsample x32 (FCN-32s) или fusion с pool4/pool3")
print("FCN-32s грубо, FCN-8s точнее (skip с pool3 + pool4)")

# DeepLab: dilated conv
print("\nDeepLab: dilated conv позволяет RF 7x7 без снижения разрешения")
print("ASPP параллельно с dilation: 1, 6, 12, 18 (rates)")


# Игрушечная dilated conv 1D
def conv1d_dil(x, k, d):
    n = len(x)
    kl = len(k)
    out = np.zeros(n - (kl - 1) * d)
    for i in range(len(out)):
        out[i] = sum(k[j] * x[i + j * d] for j in range(kl))
    return out


rng = np.random.default_rng(0)
x = rng.normal(size=20)
k = np.array([1.0, 1.0, 1.0])
for d in (1, 2, 4):
    o = conv1d_dil(x, k, d)
    print(f"dilation={d}: вывод len={len(o)}, эффективное окно={1 + 2 * d}")
