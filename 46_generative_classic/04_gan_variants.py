"""Раздел 46 — Классические генеративные модели.

Файл 4: варианты GAN.

Концепция: WGAN (Wasserstein GAN).
Заменяем JS-divergence на Wasserstein-1 distance: D становится "критиком",
не sigmoid'ом. Loss: E[D(real)] - E[D(fake)]. Для 1-липшицевости делаем
weight clipping (зажимаем веса в [-c, c]).

Концепция: WGAN-GP (Gradient Penalty).
Вместо clipping — штраф (||grad_x D(x_hat)|| - 1)^2 на интерполированных
точках x_hat = eps*real + (1-eps)*fake. Гладкое поведение, лучше сходимость.

Концепция: LSGAN.
Discriminator вывод — число (не вероятность). Loss = MSE: D учит делать
1 для real, 0 для fake. G учит D=1. Стабильнее, меньше vanishing gradient.

Концепция: Conditional GAN (cGAN).
Conditioning: подаём метку y и в G(z, y), и в D(x, y). Контролируемая
генерация: можно попросить "сгенерируй 7".

Концепция: Pix2Pix.
cGAN, где условие — целое изображение (sketch -> photo). Парные данные.
L1 loss добавляется к adversarial для пиксельной точности.

Концепция: CycleGAN.
Непарные данные (лошадь <-> зебра). Два G и два D. Cycle consistency:
G2(G1(x)) ≈ x. Identity loss дополнительно стабилизирует.
"""
import numpy as np

rng = np.random.default_rng(0)

# WGAN критик: D(x) — скаляр без sigmoid
n = 200
x_real = rng.normal(2.0, 0.5, size=n)
x_fake = rng.normal(0.0, 1.0, size=n)
w_d = 0.5
# Wasserstein loss
W = (w_d * x_real).mean() - (w_d * x_fake).mean()
print(f"WGAN критик loss = E[D(real)] - E[D(fake)] = {W:.3f}")

# Weight clipping concept
c = 0.01
w_d_clipped = np.clip(w_d, -c, c)
print(f"Weight clipping: w_d {w_d} -> {w_d_clipped} (зажат в [-{c}, {c}])")

# WGAN-GP: gradient penalty на интерполяции
eps = rng.uniform(0, 1, size=n)
x_hat = eps * x_real + (1 - eps) * x_fake
# D(x) = w_d * x -> grad_x D = w_d (1D)
grads = np.full(n, w_d)
gp = ((np.abs(grads) - 1) ** 2).mean()
print(f"Gradient penalty: ({w_d}-1)^2 = {gp:.3f}")

# LSGAN
def lsgan_d_loss(d_real, d_fake):
    return 0.5 * ((d_real - 1) ** 2).mean() + 0.5 * (d_fake ** 2).mean()


def lsgan_g_loss(d_fake):
    return 0.5 * ((d_fake - 1) ** 2).mean()


d_real = rng.uniform(0.5, 1.0, size=n)
d_fake = rng.uniform(0.0, 0.5, size=n)
print(f"LSGAN D loss = {lsgan_d_loss(d_real, d_fake):.3f}, "
      f"G loss = {lsgan_g_loss(d_fake):.3f}")

# Conditional GAN: concat label
z = rng.normal(size=(5, 4))
y_onehot = np.eye(3)[rng.integers(0, 3, size=5)]
zy = np.concatenate([z, y_onehot], axis=1)
print(f"cGAN: вход G = z (4) + y_onehot (3) = форма {zy.shape}")
print("D также получает (x, y) -> учится отличать пары (real, y) от (fake, y)")

# Pix2Pix: пара (sketch, photo)
print("\nPix2Pix: G(sketch) -> photo; D(sketch, photo) -> real/fake; +L1(photo, G(sketch))")

# CycleGAN: cycle consistency
print("CycleGAN: G_XY и G_YX; loss = adv + lambda * ||G_YX(G_XY(x)) - x||_1")
print("Identity loss: ||G_XY(y) - y||_1 — сохранение стиля")

# Игрушечная демонстрация cycle consistency на 1D
def G_xy(x):
    return 2 * x + 1


def G_yx(y):
    return (y - 1) / 2


x = rng.normal(size=5)
cycle = G_yx(G_xy(x))
print(f"Cycle consistency error на игрушечных G: max|err|={np.abs(cycle - x).max():.4f}")
