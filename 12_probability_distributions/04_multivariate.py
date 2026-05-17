"""Раздел 12 — Распределения вероятностей.

Файл 4: многомерные распределения.

Концепция 148: Многомерное нормальное MVN(mu, Sigma).
Обобщение нормального на d измерений. Полностью задаётся вектором средних
и ковариационной матрицей. Линии уровня pdf — эллипсы (эллипсоиды).

Концепция 149: Форма ковариации.
Sigma=I — изотропный шум; диагональная — независимые компоненты;
полная — корреляции. Собственные векторы Sigma задают оси эллипса.

Концепция 150: Расстояние Махаланобиса.
d_M(x, mu) = sqrt((x-mu)^T Sigma^-1 (x-mu)). "Расстояние с учётом разброса":
1 = одна стандартная единица по соответствующей оси эллипса.

Концепция 151: Сэмплирование через Cholesky.
Если L*L^T = Sigma и z ~ N(0,I), то x = mu + L*z ~ N(mu, Sigma).
Стандартный способ генерации MVN без библиотечного MVN.

Концепция 152: Смесь гауссиан (GMM) и условное MVN.
Смесь = взвешенная сумма нормалей; сэмплируем компоненту, затем точку из неё.
Условное MVN: разбиваем (x1,x2), p(x1|x2) — снова нормальное со сдвинутым
средним и уменьшенной ковариацией (формула Шура).
"""
import numpy as np

rng = np.random.default_rng(0)

# 148: MVN sampling
mu = np.array([1.0, -1.0])
Sigma = np.array([[2.0, 0.8],
                  [0.8, 1.0]])
X = rng.multivariate_normal(mu, Sigma, size=20_000)
print(f"148 MVN: emp mean={X.mean(axis=0).round(2)}, теор={mu}")
print(f"     emp cov=\n{np.cov(X.T).round(2)}")

# 149: форма Sigma — собственные числа = длины полуосей^2
eigvals, eigvecs = np.linalg.eigh(Sigma)
print(f"149 eigvals(Sigma)={eigvals.round(3)} — длины полуосей эллипса")
print(f"     главная ось ~ собств. вектору {eigvecs[:, -1].round(3)}")

# 150: Mahalanobis
inv_S = np.linalg.inv(Sigma)
diff = X - mu
mah = np.sqrt(np.einsum("ni,ij,nj->n", diff, inv_S, diff))
print(f"150 Mahalanobis: mean={mah.mean():.2f} (теор для d=2 ≈ sqrt(2)≈1.41)")

# 151: сэмплирование через Cholesky
L = np.linalg.cholesky(Sigma)
z = rng.normal(size=(20_000, 2))
X2 = mu + z @ L.T
print(f"151 Chol-sample: emp cov=\n{np.cov(X2.T).round(2)}")

# 152: GMM из двух кластеров
mu_a, mu_b = np.array([-2, 0]), np.array([3, 1])
S_a = np.eye(2) * 0.3
S_b = np.array([[0.5, -0.2], [-0.2, 0.8]])
pi = np.array([0.4, 0.6])
N = 5_000
comp = rng.choice(2, size=N, p=pi)
X_gmm = np.where(
    comp[:, None] == 0,
    rng.multivariate_normal(mu_a, S_a, size=N),
    rng.multivariate_normal(mu_b, S_b, size=N),
)
print(f"152a GMM: общее среднее={X_gmm.mean(axis=0).round(2)} (теор pi@mu={(pi[:,None]*np.stack([mu_a,mu_b])).sum(0)})")

# Условное MVN: p(x1|x2=0) для нашей Sigma
# mu1|2 = mu1 + S12/S22 * (x2 - mu2); var1|2 = S11 - S12^2/S22
x2_val = 0.0
mu1_cond = mu[0] + Sigma[0, 1] / Sigma[1, 1] * (x2_val - mu[1])
var1_cond = Sigma[0, 0] - Sigma[0, 1] ** 2 / Sigma[1, 1]
print(f"152b cond MVN: p(x1|x2=0) ~ N({mu1_cond:.3f}, {var1_cond:.3f})")
