"""Раздел 63 — Data augmentation.

Файл 2: Современные image-аугментации.

Концепция: MixUp.
Смешивает два сэмпла: x' = lam x_a + (1-lam) x_b, y' = lam y_a + (1-lam) y_b.
lam ~ Beta(alpha, alpha). Сильный регуляризатор, борется с переуверенностью.

Концепция: CutMix.
Вырезает прямоугольник из x_b и вставляет в x_a. Метка взвешивается долей вставленной площади:
y' = lam y_a + (1-lam) y_b, lam = 1 - area_cut / area_total.

Концепция: RandAugment.
Из набора N трансформаций (rotate, color, equalize, ...) выбираем M случайных и применяем
с magnitude M_int. Всего 2 гиперпараметра (N, M) — проще AutoAugment.

Концепция: AugMix.
Генерирует k цепочек случайных аугментаций, смешивает их выпуклой комбинацией
с весами Dirichlet, потом смешивает с исходным изображением с весом из Beta.
Дает большое разнообразие при контролируемом расстоянии от исходного.
"""
import numpy as np

rng = np.random.default_rng(0)

# Два "изображения" 8x8 в одном канале для простоты
x_a = rng.uniform(0, 1, (8, 8))
x_b = rng.uniform(0, 1, (8, 8))
y_a = np.array([1.0, 0.0])  # one-hot
y_b = np.array([0.0, 1.0])

# === MixUp ===
def sample_beta(alpha):
    # Beta через 2 Gamma; используем Numpy rng
    g1 = rng.gamma(alpha); g2 = rng.gamma(alpha)
    return g1 / (g1 + g2)

lam = sample_beta(0.2)
x_mix = lam * x_a + (1 - lam) * x_b
y_mix = lam * y_a + (1 - lam) * y_b
print(f"MixUp: lam={lam:.3f}, метка mixed={y_mix.round(3)}, mean(x_mix)={x_mix.mean():.3f}")

# === CutMix ===
H, W = x_a.shape
# случайный прямоугольник пропорциональный sqrt(1-lam)
lam = sample_beta(1.0)
ratio = np.sqrt(1 - lam)
cut_h, cut_w = int(H * ratio), int(W * ratio)
cy, cx = rng.integers(0, H), rng.integers(0, W)
y0 = max(0, cy - cut_h // 2); y1 = min(H, cy + cut_h // 2)
x0 = max(0, cx - cut_w // 2); x1 = min(W, cx + cut_w // 2)
x_cut = x_a.copy()
x_cut[y0:y1, x0:x1] = x_b[y0:y1, x0:x1]
area_frac = ((y1 - y0) * (x1 - x0)) / (H * W)
lam_eff = 1 - area_frac
y_cut = lam_eff * y_a + (1 - lam_eff) * y_b
print(f"CutMix: вставили {(y1-y0)}x{(x1-x0)}, lam_eff={lam_eff:.3f}, y={y_cut.round(3)}")

# === RandAugment: применить M случайных операций ===
N_ops = 4; M = 2; mag = 0.5
ops = [
    lambda x: np.clip(x + mag * 0.3, 0, 1),               # brightness
    lambda x: np.clip((x - 0.5) * (1 + mag) + 0.5, 0, 1), # contrast
    lambda x: np.rot90(x),                                # rotate90
    lambda x: np.flip(x, axis=1),                         # flip
]
chosen = rng.choice(N_ops, size=M, replace=False)
x_ra = x_a.copy()
for i in chosen:
    x_ra = ops[i](x_ra)
print(f"RandAugment: применили {M} из {N_ops}, выбраны индексы {chosen}, mean={x_ra.mean():.3f}")

# === AugMix: 3 цепочки, веса Dirichlet, итоговое смешивание ===
k = 3
dir_weights = rng.dirichlet(alpha=[1.0] * k)
chains = []
for _ in range(k):
    x_c = x_a.copy()
    depth = rng.integers(1, 4)
    for _ in range(depth):
        x_c = ops[rng.integers(N_ops)](x_c)
        if x_c.shape != x_a.shape:  # rot90 портит форму -- возвращаем
            x_c = np.resize(x_c, x_a.shape)
    chains.append(x_c)
mix_chain = sum(w * c for w, c in zip(dir_weights, chains))
m = sample_beta(1.0)
x_augmix = m * x_a + (1 - m) * mix_chain
print(f"AugMix: dir_w={dir_weights.round(3)}, m={m:.3f}, mean={x_augmix.mean():.3f}")
