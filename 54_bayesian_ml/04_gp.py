"""Раздел 54 — Bayesian ML. Файл 4: Gaussian Process regression.

Концепция: Что такое GP.
Бесконечномерное обобщение многомерной нормальной: любое конечное подмножество
точек имеет совместное гауссово распределение. Задаётся ядром k(x, x'),
которое определяет ковариацию между значениями функции в разных точках.

Концепция: RBF (Gaussian) kernel.
k(x, x') = sigma_f^2 * exp(-||x - x'||^2 / (2 * l^2)).
l — length-scale (как далеко 'влияет' одна точка), sigma_f — амплитуда.
Гладкие функции получаются.

Концепция: GP regression posterior.
Учим f на (X, y) с шумом sigma_n^2. Для тестовой точки x*:
  K = k(X, X) + sigma_n^2 I,  k* = k(X, x*)
  mu_*(x*)   = k*^T K^{-1} y
  var_*(x*)  = k(x*, x*) - k*^T K^{-1} k*.
Закрытая форма — никакого MCMC.

Концепция: Сложность.
Обращение K — O(N^3), хранение — O(N^2). Поэтому GP плохо масштабируется
на десятки тысяч точек. Решения: sparse GP (inducing points), Nyström, локальные модели.

Концепция: Применение.
Bayesian Optimization (минимизация дорогих функций), небольшие задачи регрессии
с нужной uncertainty, гиперпараметрическая оптимизация. По сути — байесовский
'универсальный регрессор' с честной epistemic uncertainty.
"""
import numpy as np

rng = np.random.default_rng(0)

# Игрушечный 1D регрессионный датасет
def true_f(x):
    return np.sin(2 * x) + 0.3 * x


X = np.array([-2.5, -1.5, -0.5, 0.0, 1.0, 2.0]).reshape(-1, 1)
y = true_f(X.flatten()) + rng.normal(scale=0.1, size=len(X))


def rbf(a, b, l=1.0, sf=1.0):
    d2 = (a - b.T) ** 2
    return sf ** 2 * np.exp(-d2 / (2 * l ** 2))


sigma_n = 0.1
l = 1.0
sf = 1.0
K = rbf(X, X, l, sf) + sigma_n ** 2 * np.eye(len(X))
K_inv = np.linalg.inv(K)

# Тестовые точки
Xs = np.linspace(-3, 3, 7).reshape(-1, 1)
Ks = rbf(X, Xs, l, sf)
Kss = rbf(Xs, Xs, l, sf)

mu_s = (Ks.T @ K_inv @ y).flatten()
cov_s = Kss - Ks.T @ K_inv @ Ks
var_s = np.diag(cov_s)

print("GP regression предсказания (x, mean ± std, true):")
for x_, m, v in zip(Xs.flatten(), mu_s, var_s):
    print(f"  x={x_:+.2f}  pred={m:+.3f} ± {np.sqrt(max(v, 0)):.3f}   true={true_f(x_):+.3f}")

# В точках, далёких от тренировочных, std больше:
far = np.array([[-5.0], [5.0]])
Ks2 = rbf(X, far, l, sf)
Kss2 = rbf(far, far, l, sf)
var_far = np.diag(Kss2 - Ks2.T @ K_inv @ Ks2)
print(f"std в дальних точках (extrapolation): {np.sqrt(var_far).round(3)}  (растёт)")
