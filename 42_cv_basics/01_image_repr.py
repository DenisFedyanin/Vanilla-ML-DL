"""Раздел 42 — Основы Computer Vision.

Файл 1: представление изображений и базовые преобразования.

Концепция: Изображение как ndarray.
Чёрно-белое — (H, W) с яркостью 0..255. Цветное — (H, W, 3) каналы RGB,
с альфой — (H, W, 4). Тип чаще uint8; для расчётов переводим в float и /255.

Концепция: RGB <-> grayscale.
Стандартная формула (luma BT.601): Y = 0.299*R + 0.587*G + 0.114*B.
Веса учитывают чувствительность глаза: к зелёному выше, к синему ниже.

Концепция: HSV.
Hue (тон, 0..360), Saturation (насыщенность), Value (яркость). Удобен
для выделения цвета: H мало меняется при изменении освещения.

Концепция: Гамма-коррекция.
y = x^gamma, где x в [0,1]. gamma<1 — осветляет тени, gamma>1 — затемняет.
Компенсирует нелинейность мониторов и восприятия.

Концепция: Гистограмма.
np.histogram(img, bins=256) — частоты яркостей. Показывает экспозицию.

Концепция: Эквализация гистограммы.
Растягиваем CDF на [0,255]: new = round((L-1) * CDF(v)). Повышает контраст:
плотные диапазоны яркости расширяются равномерно.
"""
import numpy as np

rng = np.random.default_rng(0)

# Синтетическое RGB-изображение 8x8
H, W = 8, 8
img = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
print("Форма RGB:", img.shape, "dtype:", img.dtype)

# RGB -> grayscale
gray = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.uint8)
print("Форма gray:", gray.shape, "среднее:", gray.mean().round(1))

# grayscale -> RGB (просто дублируем канал)
gray_rgb = np.stack([gray] * 3, axis=-1)
print("Gray->RGB форма:", gray_rgb.shape)

# HSV: вычислим Hue для одного пикселя вручную
r, g, b = img[0, 0] / 255.0
cmax, cmin = max(r, g, b), min(r, g, b)
delta = cmax - cmin
if delta == 0:
    hue = 0.0
elif cmax == r:
    hue = 60 * (((g - b) / delta) % 6)
elif cmax == g:
    hue = 60 * (((b - r) / delta) + 2)
else:
    hue = 60 * (((r - g) / delta) + 4)
sat = 0 if cmax == 0 else delta / cmax
val = cmax
print(f"HSV пикселя (0,0): H={hue:.0f}, S={sat:.2f}, V={val:.2f}")

# Гамма-коррекция
xf = gray.astype(np.float32) / 255.0
gamma_dark = (xf ** 2.2 * 255).astype(np.uint8)
gamma_light = (xf ** (1 / 2.2) * 255).astype(np.uint8)
print("Гамма 2.2 ср.:", gamma_dark.mean().round(1), "гамма 1/2.2 ср.:", gamma_light.mean().round(1))

# Гистограмма
hist, edges = np.histogram(gray, bins=8, range=(0, 256))
print("Гистограмма (8 бинов):", hist)

# Эквализация гистограммы вручную
full_hist, _ = np.histogram(gray, bins=256, range=(0, 256))
cdf = np.cumsum(full_hist)
cdf_norm = (cdf - cdf.min()) / (cdf.max() - cdf.min() + 1e-9)
lut = (cdf_norm * 255).astype(np.uint8)
eq = lut[gray]
print("До экв.: min/max =", gray.min(), gray.max(),
      "| после: min/max =", eq.min(), eq.max())
