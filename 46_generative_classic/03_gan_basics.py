"""Раздел 46 — Классические генеративные модели.

Файл 3: GAN основы.

Концепция: GAN.
Generator G: z -> x. Discriminator D: x -> [0, 1] (real/fake). Минимаксная
игра: min_G max_D E_x log D(x) + E_z log(1 - D(G(z))). В оптимуме D=0.5 везде,
а G моделирует распределение данных.

Концепция: Non-saturating loss for G.
В оригинале log(1 - D(G(z))) насыщается при D~0. На практике используют
-log D(G(z)) — те же gradient'ы, но без насыщения в начале обучения.

Концепция: Mode collapse.
G учится производить лишь часть мод данных (например, только одну цифру).
D перестаёт это штрафовать, если ему достаточно той моды. Решения:
mini-batch discrimination, unrolled GAN, WGAN.

Концепция: Training dynamics.
Обучение нестабильно: можно сделать k шагов D и 1 шаг G; следить за
балансом losses; нужны спец. lr и архитектура (DCGAN guidelines).
"""
import numpy as np

rng = np.random.default_rng(0)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


# Цель: смесь двух Gaussian, мы пытаемся её воспроизвести (1D)
def real_data(n):
    g = rng.choice([0, 1], size=n)
    return np.where(g == 0, rng.normal(-2, 0.5, n), rng.normal(2, 0.5, n))


# G(z) = w_g * z + b_g (линейный)
w_g, b_g = rng.normal(0, 1), 0.0
# D(x) = sigmoid(w_d * x + b_d)
w_d, b_d = rng.normal(0, 1), 0.0
lr = 0.05

n_iter = 200
batch = 64
for it in range(n_iter):
    # D step
    x_real = real_data(batch)
    z = rng.normal(size=batch)
    x_fake = w_g * z + b_g
    p_real = sigmoid(w_d * x_real + b_d)
    p_fake = sigmoid(w_d * x_fake + b_d)
    # grad ascent on log p_real + log(1 - p_fake)
    dwd = ((1 - p_real) * x_real - p_fake * x_fake).mean()
    dbd = ((1 - p_real) - p_fake).mean()
    w_d += lr * dwd
    b_d += lr * dbd
    # G step (non-saturating: max log D(G(z)))
    z = rng.normal(size=batch)
    x_fake = w_g * z + b_g
    p_fake = sigmoid(w_d * x_fake + b_d)
    # d/dw_g [log p_fake] = (1 - p_fake) * w_d * z
    dwg = ((1 - p_fake) * w_d * z).mean()
    dbg = ((1 - p_fake) * w_d).mean()
    w_g += lr * dwg
    b_g += lr * dbg

# Финальный sample
z = rng.normal(size=500)
x_gen = w_g * z + b_g
print(f"G выучил: w_g={w_g:.2f}, b_g={b_g:.2f}")
print(f"Сгенерировано: mean={x_gen.mean():.2f}, std={x_gen.std():.2f}")
print(f"Реальные: mean=0.0 (смесь -2 и 2), std≈2.06")
print("Mode collapse: при простой G=линейная мы НЕ сможем воспроизвести 2 моды.")
print("Это иллюстрирует ограничения: для бимодальности G должен быть нелинейным или MoG-style.")
