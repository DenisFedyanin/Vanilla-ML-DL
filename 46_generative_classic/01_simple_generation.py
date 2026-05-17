"""Раздел 46 — Классические генеративные модели.

Файл 1: простые методы генерации.

Концепция: Sampling из эмпирической гистограммы.
1) Считаем нормированную гистограмму данных. 2) Накапливаем CDF.
3) Берём u~U(0,1), ищем bin, где CDF(b-1)<=u<CDF(b). Так можно
семплить из любой одномерной выборки.

Концепция: GMM sampling.
Mixture of Gaussians: фитим (EM) -> для семпла выбираем компоненту
по w_k, далее N(mu_k, Sigma_k). Гибкая параметрическая модель.

Концепция: Autoregressive Bernoulli.
P(x_1..x_T) = П_t P(x_t | x_<t). Учим P(x_t | x_<t) как маленькую сетку
или логит на сумме предыдущих. Семпл — последовательно бросаем монетки.

Концепция: MLE-based generation.
Фитим параметрическое распределение (Bernoulli/Gaussian) по MLE и семплим
оттуда. Простейший baseline, без всякого скрытого пространства.
"""
import numpy as np
from sklearn.mixture import GaussianMixture

rng = np.random.default_rng(0)

# 1) Эмпирическая гистограмма
data = rng.normal(0, 1, size=500)
hist, edges = np.histogram(data, bins=20, density=True)
widths = np.diff(edges)
probs = hist * widths
probs /= probs.sum()
cdf = np.cumsum(probs)
u = rng.uniform(size=300)
idx = np.searchsorted(cdf, u)
# В каждом bin берём центр (можно ещё randomize внутри bin)
centers = 0.5 * (edges[:-1] + edges[1:])
samples_hist = centers[idx]
print(f"Sample из гистограммы: mean={samples_hist.mean():.3f}, std={samples_hist.std():.3f}")
print(f"Оригинал данных:        mean={data.mean():.3f}, std={data.std():.3f}")

# 2) GMM sampling
X = np.concatenate([
    rng.normal(-2, 0.5, size=(200, 2)),
    rng.normal(2, 0.5, size=(200, 2)),
    rng.normal([0, 3], 0.4, size=(200, 2)),
])
gm = GaussianMixture(n_components=3, random_state=0).fit(X)
samples_gmm, comp = gm.sample(300)
print(f"GMM: веса компонент = {gm.weights_.round(2)}")
print(f"Сгенерировано {len(samples_gmm)} точек, средние по компонентам:")
for k in range(3):
    print(f"  компонента {k}: mean={gm.means_[k].round(2)}, sample-mean={samples_gmm[comp == k].mean(axis=0).round(2)}")

# 3) Autoregressive Bernoulli
T = 10
# Истинная "цепочка": p(x_t=1) = sigmoid(0.5 + 1.0 * x_{t-1})
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


N = 300
seqs = np.zeros((N, T), dtype=int)
seqs[:, 0] = rng.uniform(size=N) < 0.5
for t in range(1, T):
    p = sigmoid(0.5 + 1.0 * seqs[:, t - 1])
    seqs[:, t] = rng.uniform(size=N) < p
print("AR Bernoulli: средняя x_t по позиции t =", seqs.mean(axis=0).round(2))
print("Видно, что после x_{t-1}=1 шансы x_t=1 растут — автокорреляция")

# 4) MLE-Gaussian baseline
mu_hat, sigma_hat = data.mean(), data.std()
gen_mle = rng.normal(mu_hat, sigma_hat, size=300)
print(f"MLE Gaussian sampling: mean={gen_mle.mean():.3f}, std={gen_mle.std():.3f}")
