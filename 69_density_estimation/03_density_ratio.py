"""Раздел 69 — Density estimation.

Файл 3: Density ratio estimation, copulas.

Концепция: Density ratio estimation (DRE).
Часто нужна не плотность, а ОТНОШЕНИЕ r(x) = p(x) / q(x). Применения:
covariate shift correction, importance sampling, GAN-like обучение, KL-дивергенция.

Концепция: DRE через классификацию.
Обозначим p-сэмплы меткой 1, q-сэмплы меткой 0 и обучим классификатор P(y=1|x).
Тогда:
  p(x) / q(x) = P(y=1|x) / P(y=0|x) * P(y=0) / P(y=1).
Если классов поровну, P(y=0)/P(y=1) = 1 и отношение = sigmoid_logit. Это и есть
'трюк density ratio через классификатор' (LSIF, ulSIF, NN-DRE).

Концепция: KL и JS через DRE.
KL(p || q) = E_p[log r(x)]. Если есть r, легко считается на p-сэмплах.

Концепция: Copula (концепт).
Многомерное распределение можно разбить на маржинальные F_1,..,F_d (каждая 1D)
и копулу C, описывающую структуру зависимости: F(x) = C(F_1(x_1),..,F_d(x_d)).
Sklar's theorem. Популярны Gaussian copula, t-copula, Archimedean (Clayton, Gumbel).
Применение: моделирование зависимостей с гибкими маржиналами.

Демо: оцениваем p(x)/q(x) логистической регрессией и сравниваем с истинным
(p=N(0,1), q=N(1,1.5)). На сетке точек.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(0)

# === истинные p и q ===
def pdf_normal(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

n = 2000
Xp = rng.normal(0, 1, size=(n, 1))      # p = N(0,1)
Xq = rng.normal(1, 1.5, size=(n, 1))    # q = N(1, 1.5)

# === обучаем классификатор p vs q ===
X = np.concatenate([Xp, Xq])
y = np.concatenate([np.ones(n), np.zeros(n)])
clf = LogisticRegression().fit(X, y)

# === считаем r_hat(x) = P(1|x) / P(0|x) ===
grid = np.linspace(-4, 6, 11).reshape(-1, 1)
prob1 = clf.predict_proba(grid)[:, 1]
r_hat = prob1 / (1 - prob1 + 1e-12)
r_true = pdf_normal(grid.ravel(), 0, 1) / pdf_normal(grid.ravel(), 1, 1.5)

print(f"{'x':>5} {'r_true':>10} {'r_hat':>10}")
for x_, rt, rh in zip(grid.ravel(), r_true, r_hat):
    print(f"{x_:>5.1f} {rt:>10.3f} {rh:>10.3f}")

err = np.abs(np.log(r_hat + 1e-12) - np.log(r_true + 1e-12)).mean()
print(f"\nMean |log r_hat - log r_true| = {err:.3f}")

# === KL(p || q) через DRE ===
# KL ~ E_p[log r(x)] = mean логарифма r_hat по p-сэмплам
prob1_p = clf.predict_proba(Xp)[:, 1]
log_r_p = np.log(prob1_p + 1e-12) - np.log(1 - prob1_p + 1e-12)
kl_dre = log_r_p.mean()

# теоретическое KL(N(0,1) || N(1, 1.5)):
mu_p, s_p, mu_q, s_q = 0, 1, 1, 1.5
kl_true = np.log(s_q / s_p) + (s_p ** 2 + (mu_p - mu_q) ** 2) / (2 * s_q ** 2) - 0.5
print(f"\nKL(p || q): через DRE = {kl_dre:.3f}, аналитически = {kl_true:.3f}")

# === Copula (концепт + простой sample) ===
# Создадим 2D зависимость через Gaussian copula:
# 1) z ~ N(0, Sigma) с Sigma = [[1, 0.8],[0.8, 1]]
# 2) u = Phi(z) — равномерные с зависимостью
# 3) маргиналы: x1 ~ Exp(1) <- F^{-1}_exp(u1); x2 ~ N(0,1) <- Phi^{-1}(u2) = z_2
from math import erf
def Phi(z):
    return 0.5 * (1 + np.array([erf(zi / np.sqrt(2)) for zi in z.ravel()])).reshape(z.shape)

Sigma = np.array([[1.0, 0.8], [0.8, 1.0]])
L = np.linalg.cholesky(Sigma)
Z = rng.normal(size=(1000, 2)) @ L.T
U = Phi(Z)
x1 = -np.log(1 - U[:, 0])  # маргиналь Exp(1)
x2 = Z[:, 1]               # маргиналь N(0,1)
corr_orig = np.corrcoef(Z.T)[0, 1]
corr_marg = np.corrcoef(x1, x2)[0, 1]
print(f"\nGaussian copula: задано rho={Sigma[0,1]:.2f}, "
      f"эмпирически rho(z)={corr_orig:.3f}, rho(x_with_diff_margins)={corr_marg:.3f}")
print("Копула отделяет 'форму' зависимости от маржинальных распределений.")
