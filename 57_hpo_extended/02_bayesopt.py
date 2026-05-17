"""Раздел 57 — HPO.

Файл 2: Байесовская оптимизация (BO).

Концепция: GP-суррогат.
Гауссов процесс — априорное распределение над функциями. По наблюденным точкам
дает posterior mean mu(x) и std sigma(x). Используется как дешевая модель дорогой f.

Концепция: Expected Improvement (EI).
EI(x) = E[max(0, f_best - f(x))] при минимизации. Балансирует exploit (низкая mu)
и explore (высокая sigma). Аналитическая формула через стандартный нормальный CDF/PDF.

Концепция: Upper Confidence Bound (UCB / LCB).
LCB(x) = mu(x) - kappa * sigma(x). Минимизируем — выбираем точки с низким mu или большой sigma.
Параметр kappa регулирует степень исследования.

Концепция: Probability of Improvement (PI).
PI(x) = P(f(x) < f_best). Простая, но склонна к чрезмерному exploit при низком sigma.

Концепция: Цикл BO.
1) Фитуем GP на (X,y). 2) Максимизируем acquisition на сетке/случайных точках.
3) Вычисляем f в новой точке. 4) Повторяем.
"""
import math
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

rng = np.random.default_rng(0)

# Дорогая 1D-функция (минимизируем)
f = lambda x: np.sin(3 * x) + 0.3 * (x - 0.5) ** 2
X_grid = np.linspace(0, 2, 200).reshape(-1, 1)

# Стандартный нормальный PDF/CDF без scipy
norm_pdf = lambda z: np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
norm_cdf = lambda z: 0.5 * (1.0 + np.vectorize(lambda v: math.erf(v / np.sqrt(2)))(z))

def ei(mu, sigma, y_best, xi=0.01):
    sigma = np.maximum(sigma, 1e-9)
    z = (y_best - mu - xi) / sigma
    return (y_best - mu - xi) * norm_cdf(z) + sigma * norm_pdf(z)

def lcb(mu, sigma, kappa=2.0):
    return -(mu - kappa * sigma)  # вернем "выше = лучше"

def pi(mu, sigma, y_best, xi=0.01):
    sigma = np.maximum(sigma, 1e-9)
    return norm_cdf((y_best - mu - xi) / sigma)

# Стартовые точки
X = rng.uniform(0, 2, size=(3, 1))
y = f(X).ravel()

kernel = ConstantKernel(1.0) * RBF(length_scale=0.3)
for it in range(8):
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True).fit(X, y)
    mu, sigma = gp.predict(X_grid, return_std=True)
    acq = ei(mu, sigma, y.min())
    x_next = X_grid[np.argmax(acq)]
    X = np.vstack([X, x_next])
    y = np.append(y, f(x_next).item())

print(f"BO+EI: лучшее x={X[np.argmin(y)].item():.3f}, f={y.min():.4f}")

# Сравним acquisition'ы на одном GP в финале
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True).fit(X, y)
mu, sigma = gp.predict(X_grid, return_std=True)
print(f"EI argmax x = {X_grid[np.argmax(ei(mu, sigma, y.min()))].item():.3f}")
print(f"LCB argmax x = {X_grid[np.argmax(lcb(mu, sigma))].item():.3f}")
print(f"PI argmax x = {X_grid[np.argmax(pi(mu, sigma, y.min()))].item():.3f}")
