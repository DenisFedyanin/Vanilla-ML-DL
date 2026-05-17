"""Раздел 47 — Современные генеративные модели.

Файл 3: DDPM (Denoising Diffusion Probabilistic Models).

Концепция: Forward diffusion.
Постепенно добавляем шум: q(x_t | x_{t-1}) = N(sqrt(1-beta_t) x_{t-1}, beta_t I).
В пределе t -> T получаем чистый шум N(0, I).

Концепция: Closed-form для x_t из x_0.
alpha_t = 1 - beta_t, alpha_bar_t = П_{s<=t} alpha_s.
q(x_t | x_0) = N(sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I).
То есть x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) * eps, eps~N(0,I).

Концепция: Reverse process.
Учим сеть eps_theta(x_t, t) предсказывать шум. Loss:
L = E_{t, x_0, eps} ||eps - eps_theta(sqrt(a_bar) x_0 + sqrt(1-a_bar) eps, t)||^2.

Концепция: Denoising step.
x_{t-1} = (1/sqrt(alpha_t)) (x_t - (beta_t / sqrt(1 - alpha_bar_t)) eps_theta)
       + sigma_t * z, z ~ N(0, I).

Концепция: Noise schedule.
Linear beta от 1e-4 до 0.02 (DDPM). Косинусный schedule сохраняет больше
информации в начале — лучше качество (improved DDPM).
"""
import numpy as np

rng = np.random.default_rng(0)
T = 100
# Linear beta schedule
betas = np.linspace(1e-4, 0.02, T)
alphas = 1 - betas
alpha_bar = np.cumprod(alphas)

# Игрушечные данные: 1D смесь Gaussian
def real_x0(n):
    g = rng.choice([0, 1], size=n)
    return np.where(g == 0, rng.normal(-2, 0.5, n), rng.normal(2, 0.5, n))


x0 = real_x0(8)
print("x0 (8 точек):", x0.round(2))

# Forward: x_t из x_0 для разных t
for t in [0, 10, 50, 99]:
    a_bar = alpha_bar[t]
    eps = rng.normal(size=x0.shape)
    xt = np.sqrt(a_bar) * x0 + np.sqrt(1 - a_bar) * eps
    print(f"t={t}: alpha_bar={a_bar:.3f}, x_t std={xt.std():.2f}")
print("При t=T x_t -> N(0, I) — чистый шум")

# Oracle denoiser: знаем eps -> можем сделать "идеальный" reverse step
# Симулируем reverse для одного семпла
N = 500
x = rng.normal(size=N)  # начнём с шума
# Будем использовать "знание" o x0 нашего синтетического процесса:
# для демонстрации сравним два варианта. Истинного x0 у нас нет в реальности,
# но мы убедимся, что forward + reverse с истинным шумом возвращает x0.
x0_test = real_x0(N)
t = 50
a_bar = alpha_bar[t]
eps_true = rng.normal(size=N)
x_t = np.sqrt(a_bar) * x0_test + np.sqrt(1 - a_bar) * eps_true
# Восстановим x0 из x_t и истинного eps:
x0_recovered = (x_t - np.sqrt(1 - a_bar) * eps_true) / np.sqrt(a_bar)
print(f"Verify: max|x0 - x0_recovered| = {np.abs(x0_test - x0_recovered).max():.6f}")

# Один reverse step DDPM (oracle: eps_theta = eps_true)
def reverse_step(x_t, eps_pred, t):
    a_t = alphas[t]
    a_bar_t = alpha_bar[t]
    beta_t = betas[t]
    mean = (1 / np.sqrt(a_t)) * (x_t - (beta_t / np.sqrt(1 - a_bar_t)) * eps_pred)
    if t == 0:
        return mean
    sigma = np.sqrt(beta_t)
    return mean + sigma * rng.normal(size=x_t.shape)


# Reverse loop с oracle на одной траектории
x0_single = np.array([1.5])
x_traj = [x0_single.copy()]
# forward
for t in range(T):
    eps = rng.normal(size=1)
    x_new = np.sqrt(alphas[t]) * x_traj[-1] + np.sqrt(betas[t]) * eps
    x_traj.append(x_new)
print(f"После forward (T={T} шагов) x_T = {x_traj[-1][0]:.2f} (должно быть ~N(0,1))")
print("В реальном DDPM мы учим eps_theta нейронкой; здесь продемонстрировали схему.")
