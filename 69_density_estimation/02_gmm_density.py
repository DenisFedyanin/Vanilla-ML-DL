"""Раздел 69 — Density estimation.

Файл 2: GMM-based density estimation.

Концепция: GMM как density model.
p(x) = sum_k pi_k * N(x | mu_k, Sigma_k). Подгоняем EM-алгоритмом. Полученная модель
даёт ЯВНУЮ непрерывную плотность, в отличие от KDE — и легко семплируется.

Концепция: Score samples.
log p(x) = log sum_k pi_k * N(x | mu_k, Sigma_k). sklearn-GaussianMixture.score_samples
возвращает log-density для каждой точки. Сравнивая лог-вероятности train vs test,
оцениваем переобучение.

Концепция: Order selection (сколько компонент K выбрать?).
- BIC = -2 log L + p log n, где p — число параметров, n — число точек. Чем меньше, тем лучше.
- AIC = -2 log L + 2 p. Менее строго штрафует сложность.
- Cross-validated log-likelihood.

Концепция: BIC и AIC.
BIC сильнее штрафует сложность, поэтому выбирает более простые модели; асимптотически
сходится к истинной модели (если она в семействе). AIC лучше для предсказательных целей.

Демо: на смеси из 3 гауссиан перебираем K=1..5 и выбираем по BIC.
"""
import numpy as np
from sklearn.mixture import GaussianMixture

rng = np.random.default_rng(0)

# === истинная смесь 3 компонент в 2D ===
true_means = np.array([[0, 0], [4, 4], [-3, 3]])
true_cov = 0.5 * np.eye(2)
n_per = 200
X = np.concatenate([rng.multivariate_normal(m, true_cov, n_per) for m in true_means])

# === перебираем K ===
results = []
for K in range(1, 6):
    gmm = GaussianMixture(n_components=K, covariance_type="full",
                         random_state=0, n_init=3).fit(X)
    bic = gmm.bic(X)
    aic = gmm.aic(X)
    ll = gmm.score(X) * len(X)  # сумма log-likelihood
    results.append((K, bic, aic, ll))
    print(f"K={K}: BIC={bic:8.1f}, AIC={aic:8.1f}, sum_logL={ll:8.1f}")

best_K = min(results, key=lambda r: r[1])[0]
print(f"\nЛучшее K по BIC: {best_K} (истинное = 3)")

# === обучаем итоговую модель и смотрим параметры ===
gmm = GaussianMixture(n_components=best_K, covariance_type="full",
                     random_state=0, n_init=3).fit(X)
print(f"Веса найденных компонент: {np.round(gmm.weights_, 3)}")
print(f"Средние компонент:\n{np.round(gmm.means_, 2)}")
print(f"Истинные средние:\n{true_means}")

# === score_samples: log-density новых точек ===
test_pts = np.array([[0, 0], [4, 4], [-3, 3], [10, 10]])
log_dens = gmm.score_samples(test_pts)
print(f"\nlog p(x) в точках:")
for p, ld in zip(test_pts, log_dens):
    print(f"  x={p.tolist()}: log p = {ld:.2f}, p ~ {np.exp(ld):.4f}")

# === семплирование из обученной модели ===
samples, comp = gmm.sample(500)
print(f"\nСемплов сгенерировано: {len(samples)}, "
      f"распределение по компонентам: {np.bincount(comp)}")
