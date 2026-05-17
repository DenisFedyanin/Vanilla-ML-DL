"""Раздел 32 — Регуляризация (расширенно). Файл 2: на входе и метках.

Концепция 365: Label smoothing.
Метку one-hot заменяем на (1-eps)*onehot + eps/K — все классы получают
небольшую долю. Регуляризует softmax от перекоса в один класс.

Концепция 366: MixUp.
Берём две пары (x1,y1),(x2,y2) и обучаем на их линейной комбинации:
x = lam*x1 + (1-lam)*x2, y = lam*y1+(1-lam)*y2, lam~Beta(alpha,alpha).
Гладит границу решения, повышает робастность.

Концепция 367: CutMix.
В картинку x1 вклеиваем прямоугольник из x2; метка — взвешенная сумма
пропорционально площади. Локальный аналог MixUp.

Концепция 368: CutOut.
В картинке вырезаем случайный прямоугольник (заполняем 0). Простая
аугментация-регуляризация.

Концепция 369: RandAugment (концепт).
Список из M аугментаций случайной "силой". Из N выбирается случайно K.
Не требует подбора каждого набора отдельно.
"""
import numpy as np

rng = np.random.default_rng(0)

# === 365: Label smoothing ===
def label_smooth(y_onehot, eps):
    K = y_onehot.shape[-1]
    return (1 - eps) * y_onehot + eps / K

y = np.eye(4)[2]
print(f"365 Label smoothing eps=0.1: {y.tolist()} -> {label_smooth(y, 0.1).round(3).tolist()}")

# === 366: MixUp ===
def mixup(x1, x2, y1, y2, alpha=0.2, rng=rng):
    lam = float(rng.beta(alpha, alpha))
    return lam * x1 + (1 - lam) * x2, lam * y1 + (1 - lam) * y2, lam

x1 = rng.standard_normal((3, 3))     # типа картинка 3x3
x2 = rng.standard_normal((3, 3))
yh1, yh2 = np.eye(3)[0], np.eye(3)[2]
x_mix, y_mix, lam = mixup(x1, x2, yh1, yh2)
print(f"366 MixUp lam={lam:.3f}: y_mix = {y_mix.round(3).tolist()} (не one-hot)")

# === 367: CutMix ===
def cutmix(x1, x2, y1, y2, rng=rng):
    H, W = x1.shape
    lam = float(rng.beta(1.0, 1.0))
    cut = int(round(np.sqrt(1 - lam) * H))
    cx, cy = rng.integers(0, H), rng.integers(0, W)
    x0 = max(0, cx - cut // 2); x1_ = min(H, cx + cut // 2)
    y0 = max(0, cy - cut // 2); y1_ = min(W, cy + cut // 2)
    new_x = x1.copy()
    new_x[x0:x1_, y0:y1_] = x2[x0:x1_, y0:y1_]
    area_frac = (x1_ - x0) * (y1_ - y0) / (H * W)
    new_y = (1 - area_frac) * y1 + area_frac * y2
    return new_x, new_y, area_frac

x_cm, y_cm, area = cutmix(x1, x2, yh1, yh2)
print(f"367 CutMix: вклеено {area * 100:.0f}% площади, y_mix={y_cm.round(3).tolist()}")

# === 368: CutOut ===
def cutout(x, size, rng=rng):
    H, W = x.shape
    cx, cy = rng.integers(0, H), rng.integers(0, W)
    x0 = max(0, cx - size // 2); x1_ = min(H, cx + size // 2)
    y0 = max(0, cy - size // 2); y1_ = min(W, cy + size // 2)
    out = x.copy(); out[x0:x1_, y0:y1_] = 0
    return out

img = rng.standard_normal((6, 6))
img_co = cutout(img, size=3)
print(f"368 CutOut: количество занулённых пикселей = {(img_co == 0).sum() - (img == 0).sum()}")

# === 369: RandAugment (концепт) ===
ops = ["identity", "shift", "rotate", "color-jitter", "blur", "sharpen", "contrast", "brightness"]
N, M = 2, 5     # из всех ops выберем N с силой M (0..10)
chosen = rng.choice(ops, size=N, replace=False)
print(f"369 RandAugment(N={N}, M={M}): выбрано {chosen.tolist()}")
print("    Каждая операция применяется с фиксированной 'силой' M.")
