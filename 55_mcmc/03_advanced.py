"""Раздел 55 — MCMC. Файл 3: Продвинутые методы и VI.

Концепция: Slice sampling.
Чтобы сэмплировать x ~ p(x): вводим auxiliary u ~ U(0, p(x)) и поочерёдно:
1) u ~ U(0, p(x_current));
2) x ~ U({x : p(x) > u}) — горизонтальный 'слой'.
Часто не требует тюнинга шага, в отличие от MH.

Концепция: Hamiltonian Monte Carlo (HMC).
Берём градиент log p и трактуем -log p как 'потенциальную энергию'. Вводим
импульс p ~ N(0, M), интегрируем гамильтониан leapfrog-шагами, делаем
MH-accept на основе сохранения энергии. В разы быстрее random-walk MH в высоких
размерностях, потому что 'движется по направлению'.

Концепция: NUTS (No U-Turn Sampler).
Авто-настройка длины траектории HMC: продолжаем шагать, пока траектория не
'разворачивается обратно' (U-turn detection). Без ручной настройки числа шагов.

Концепция: Parallel tempering.
Несколько цепей с разными 'температурами' beta_i: target_i(x) ∝ p(x)^{beta_i}.
'Горячие' цепи легко преодолевают барьеры, периодически обмениваются с
'холодными'. Помогает с мультимодальностью.

Концепция: Variational Inference (VI).
Вместо сэмплирования ищем q*(x) из семейства Q, минимизируя KL(q || p).
Эквивалентно максимизации ELBO = E_q[log p(x, z)] - E_q[log q(z)].
Mean-field: q(z) = ∏ q_i(z_i). BBVI: градиент ELBO через reparam или REINFORCE-trick.
"""
import numpy as np

rng = np.random.default_rng(0)

# === Slice sampling для p(x) = N(0,1) ===
def p(x):
    return np.exp(-0.5 * x ** 2)


N = 5000
x = 0.0
xs = np.zeros(N)
w = 1.0
for t in range(N):
    u = rng.uniform(0, p(x))
    # Расширяем интервал, пока p > u по краям
    lo, hi = x - w, x + w
    while p(lo) > u:
        lo -= w
    while p(hi) > u:
        hi += w
    while True:
        x_new = rng.uniform(lo, hi)
        if p(x_new) > u:
            x = x_new
            break
        if x_new < x:
            lo = x_new
        else:
            hi = x_new
    xs[t] = x

print(f"Slice sampling N(0,1): mean={xs.mean():+.3f}, std={xs.std():.3f}")

# === Mean-field VI: подгоняем q(x) = N(mu, sigma^2) к p(x) = N(2, 1.5^2) ===
# Целевая (известная): log p(x) = -0.5*((x-2)/1.5)^2 - log(1.5*sqrt(2*pi))
# Для нормального q ELBO считается аналитически:
# ELBO = -KL(q || p)
# Найдём оптимальные mu, sigma градиентным подъёмом по ELBO
def neg_elbo(mu, sig, S=2000):
    eps = rng.normal(size=S)
    z = mu + sig * eps
    log_p = -0.5 * ((z - 2) / 1.5) ** 2 - np.log(1.5 * np.sqrt(2 * np.pi))
    log_q = -0.5 * ((z - mu) / sig) ** 2 - np.log(sig * np.sqrt(2 * np.pi))
    return -(log_p - log_q).mean()


# Reparam-trick градиенты численно
mu, sig = 0.0, 1.0
lr = 0.05
for step in range(400):
    eps = rng.normal(size=500)
    z = mu + sig * eps
    log_p_grad_mu = -((z - 2) / 1.5 ** 2)             # d log p / d z
    # d ELBO / d mu = E[ d log p / d z * d z/d mu - d log q / d mu (но d log q / d mu = -d log q / d z)]
    # Используем: ELBO = E_eps[log p(mu+sig*eps) - log q]; grad по mu = E[d log p / d z * 1]; minus -d log q d mu
    g_mu = log_p_grad_mu.mean() - (-(z - mu) / sig ** 2).mean()
    g_sig = (log_p_grad_mu * eps).mean() - (-(z - mu) * eps / sig ** 2).mean() + 1.0 / sig  # +entropy term
    mu = mu + lr * g_mu
    sig = max(0.05, sig + lr * g_sig)

print(f"VI результат: q=N({mu:.3f}, {sig:.3f}^2)   target N(2, 1.5^2)")

# Parallel tempering и HMC — описаны в docstring (концепты), без полной реализации
print("Slice/HMC/NUTS/PT/VI — описаны в docstring; полные реализации опускаем по объёму.")
