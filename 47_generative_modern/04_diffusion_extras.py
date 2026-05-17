"""Раздел 47 — Современные генеративные модели.

Файл 4: расширения diffusion.

Концепция: DDIM (Denoising Diffusion Implicit Models).
Делает reverse process детерминированным. Семплирует за гораздо меньшее
число шагов (50 вместо 1000) с сохранением качества. Формула без
стохастического члена + поворот по noise.

Концепция: Classifier-free guidance.
Учим одну сеть с условиями y и без (с вероятностью drop_y=0.1 во время
обучения). На семплинге: eps = (1+w) eps(x, y) - w eps(x, null).
Параметр w контролирует силу следования condition.

Концепция: Latent Diffusion / Stable Diffusion.
Сначала VAE-encoder кодирует x в латент z (например, 4x64x64 вместо 3x512x512).
Диффузия в z. Декодер VAE из z обратно в x. Дёшево и быстро.

Концепция: Score-based models.
Учим s_theta(x) ≈ grad_x log p(x) (score function). Семплинг через Langevin
dynamics: x_{k+1} = x_k + eta * s + sqrt(2 eta) * z. Эквивалентно diffusion
при правильном параметре.

Концепция: Consistency models.
Прямо учат f(x_t, t) -> x_0 за один шаг. Используют distillation diffusion-модели.
"""
import numpy as np

rng = np.random.default_rng(0)

# DDIM: детерминированный reverse шаг
T = 50
betas = np.linspace(1e-4, 0.02, T)
alphas = 1 - betas
alpha_bar = np.cumprod(alphas)


def ddim_step(x_t, eps_pred, t, t_prev):
    a_bar_t = alpha_bar[t]
    a_bar_p = alpha_bar[t_prev] if t_prev >= 0 else 1.0
    x0_pred = (x_t - np.sqrt(1 - a_bar_t) * eps_pred) / np.sqrt(a_bar_t)
    dir_xt = np.sqrt(1 - a_bar_p) * eps_pred
    return np.sqrt(a_bar_p) * x0_pred + dir_xt


x_T = rng.normal(size=4)
# Прыжки по subset шагов (skip=10)
schedule = list(range(T - 1, -1, -10))
x_t = x_T.copy()
eps_dummy = rng.normal(size=4)  # представим, что сеть всегда возвращает это
for i, t in enumerate(schedule):
    t_prev = schedule[i + 1] if i + 1 < len(schedule) else -1
    x_t = ddim_step(x_t, eps_dummy, t, t_prev)
print(f"DDIM семплинг за {len(schedule)} шагов вместо {T}: x = {x_t.round(2)}")

# Classifier-free guidance
def cfg(eps_cond, eps_uncond, w=3.0):
    return (1 + w) * eps_cond - w * eps_uncond


eps_c = np.array([0.3])
eps_u = np.array([0.1])
for w in (0.0, 1.0, 3.0, 7.5):
    print(f"CFG w={w}: eps = {cfg(eps_c, eps_u, w)[0]:.3f}")
print("w=0 — игнор condition, w большое — сильное следование")

# Latent diffusion: encoder -> z, diffusion в z, decoder -> x
print("\nLatent diffusion:")
print("  image 3x512x512 -> VAE encoder -> z 4x64x64 (64x экономия памяти)")
print("  диффузия в z (быстро); decoder восстанавливает картинку")

# Score-based: Langevin dynamics для семпла из p(x) ~ exp(-x^2/2)
# Истинный score = -x. Langevin: x_{k+1} = x_k - eta * x_k + sqrt(2 eta) z
x = rng.normal(size=100) * 5.0
eta = 0.1
for _ in range(100):
    x = x - eta * x + np.sqrt(2 * eta) * rng.normal(size=100)
print(f"\nScore-based Langevin: mean={x.mean():.2f}, std={x.std():.2f} (цель N(0,1))")

# Consistency model: один forward — сразу x_0
print("\nConsistency model: f(x_t, t) -> x_0 за 1 шаг (или few-step)")
print("Учат self-consistency: f(x_t, t) = f(x_t', t') для пар на одной траектории")
