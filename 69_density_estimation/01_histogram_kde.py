"""Раздел 69 — Density estimation.

Файл 1: Гистограмма и KDE.

Концепция: Histogram density.
Разбиваем диапазон на B бинов ширины h, считаем количество точек в каждом и нормируем
так, чтобы суммарная площадь была 1: f_hat(x) = count(bin_of_x) / (n * h).
Просто и быстро, но даёт ступенчатую оценку, зависит от выбора границ бинов.

Концепция: Kernel Density Estimation (KDE).
f_hat(x) = (1 / (n h)) sum_{i=1..n} K((x - x_i) / h),
где K — ядро (обычно Gaussian K(u) = 1/sqrt(2pi) exp(-u^2/2)),
h — bandwidth. Сглаживание гистограммы: каждая точка превращается в маленький бугорок.

Концепция: Выбор bandwidth.
Маленькое h -> overfitting (узкие пики на каждой точке).
Большое h -> oversmoothing (теряется структура).

Концепция: Silverman's rule of thumb.
h = 1.06 * sigma * n^{-1/5}, где sigma — стандартное отклонение выборки.
Оптимально для нормального распределения; на практике часто берут с modifier.

Концепция: Scott's rule.
h = sigma * n^{-1/5}. Похоже на Silverman, чуть меньше bandwidth.

Концепция: Multivariate KDE.
Та же формула с произведением одномерных ядер либо с h как ковариационной матрицей.
"""
import numpy as np

rng = np.random.default_rng(0)

# === bimodal sample: смесь двух нормальных ===
n = 500
mix = rng.uniform(size=n) < 0.5
samples = np.where(mix, rng.normal(-2, 1, n), rng.normal(2, 0.7, n))

# === Histogram density ===
counts, edges = np.histogram(samples, bins=30)
bin_w = edges[1] - edges[0]
hist_density = counts / (n * bin_w)
print(f"Histogram: bins=30, bin_width={bin_w:.3f}")
print(f"  Сумма площадей = {hist_density.sum() * bin_w:.3f} (должно быть 1)")
print(f"  Max density value = {hist_density.max():.3f}")

# === KDE (Gaussian kernel) ===
def gauss_kde(x, samples, h):
    n = len(samples)
    diff = (x[:, None] - samples[None, :]) / h
    kvals = np.exp(-0.5 * diff ** 2) / np.sqrt(2 * np.pi)
    return kvals.sum(1) / (n * h)

sigma = samples.std()
h_silv = 1.06 * sigma * n ** (-1 / 5)
h_scott = sigma * n ** (-1 / 5)
print(f"\nsigma={sigma:.3f}")
print(f"Silverman h = {h_silv:.3f}")
print(f"Scott     h = {h_scott:.3f}")

grid = np.linspace(-6, 6, 200)
dens_silv = gauss_kde(grid, samples, h_silv)
dens_scott = gauss_kde(grid, samples, h_scott)
dens_small = gauss_kde(grid, samples, 0.1)
dens_big = gauss_kde(grid, samples, 2.0)
print(f"\nИнтеграл KDE (сумма * dx, должно ~1):")
dx = grid[1] - grid[0]
for name, d in [("Silverman", dens_silv), ("Scott", dens_scott),
                ("h=0.1 (overfit)", dens_small), ("h=2.0 (oversmooth)", dens_big)]:
    print(f"  {name:25s}: integral={d.sum() * dx:.3f}, max={d.max():.3f}")

# Истинная плотность для сравнения:
true_dens = 0.5 * np.exp(-0.5 * ((grid + 2) / 1.0) ** 2) / np.sqrt(2 * np.pi) + \
            0.5 * np.exp(-0.5 * ((grid - 2) / 0.7) ** 2) / np.sqrt(2 * np.pi * 0.49)
for name, d in [("Silverman", dens_silv), ("Scott", dens_scott),
                ("h=0.1", dens_small), ("h=2.0", dens_big)]:
    mse = ((d - true_dens) ** 2).mean()
    print(f"  MSE(KDE {name}, true) = {mse:.5f}")

# === Multivariate KDE (2D) ===
def gauss_kde_md(X_grid, samples, h):
    """Изотропное Gaussian ядро."""
    n, d = samples.shape
    diff = X_grid[:, None, :] - samples[None, :, :]
    sq = (diff ** 2).sum(-1) / (h ** 2)
    kvals = np.exp(-0.5 * sq) / (2 * np.pi) ** (d / 2)
    return kvals.sum(1) / (n * h ** d)

X2 = rng.normal(size=(200, 2))
gg = np.array([[0, 0], [2, 2], [-2, 2]])
print(f"\n2D KDE в точках {gg.tolist()}: dens = {np.round(gauss_kde_md(gg, X2, h=0.5), 3)}")
