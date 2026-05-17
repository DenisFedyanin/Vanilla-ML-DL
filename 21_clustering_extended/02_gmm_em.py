"""Раздел 21 — Расширенная кластеризация. Файл 2: GMM и алгоритм EM.

Концепция 625: Gaussian Mixture Model (GMM).
Данные моделируются как смесь K нормальных распределений:
p(x) = sum_k pi_k * N(x | mu_k, Sigma_k). Каждая точка имеет «мягкую»
принадлежность кластерам.

Концепция 626: EM-алгоритм — E-шаг.
При фиксированных параметрах вычисляем «ответственности»:
gamma_{ik} = pi_k * N(x_i | mu_k, Sigma_k) / sum_j (pi_j * N(x_i | mu_j, Sigma_j)).
Это вероятность принадлежности точки i кластеру k.

Концепция 627: EM-алгоритм — M-шаг.
По gamma пересчитываем параметры:
  N_k = sum_i gamma_{ik}
  mu_k = (1/N_k) * sum_i gamma_{ik} * x_i
  Sigma_k = (1/N_k) * sum_i gamma_{ik} * (x_i-mu_k)(x_i-mu_k)^T
  pi_k = N_k / N

Концепция 628: Лог-правдоподобие монотонно растёт.
EM гарантирует non-decreasing log-likelihood. Это удобный критерий сходимости.
"""
import numpy as np
from sklearn.datasets import make_blobs

rng = np.random.default_rng(0)
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=0)
N, D = X.shape
K = 3

# === 625: инициализация параметров GMM ===
init_idx = rng.choice(N, K, replace=False)
mu = X[init_idx].copy()
Sigma = np.array([np.eye(D) for _ in range(K)])
pi = np.full(K, 1 / K)

def gauss_pdf(x, m, S):
    diff = x - m
    inv = np.linalg.inv(S)
    det = np.linalg.det(S)
    n = -0.5 * np.einsum("ni,ij,nj->n", diff, inv, diff)
    return np.exp(n) / np.sqrt((2 * np.pi) ** D * det)

# === 626-628: EM-цикл ===
prev_ll = -np.inf
for it in range(30):
    # E-шаг
    probs = np.column_stack([pi[k] * gauss_pdf(X, mu[k], Sigma[k]) for k in range(K)])
    ll = np.log(probs.sum(axis=1) + 1e-300).sum()
    gamma = probs / probs.sum(axis=1, keepdims=True)
    # M-шаг
    Nk = gamma.sum(axis=0)
    mu = (gamma.T @ X) / Nk[:, None]
    Sigma = np.zeros((K, D, D))
    for k in range(K):
        diff = X - mu[k]
        Sigma[k] = (gamma[:, k, None] * diff).T @ diff / Nk[k]
        Sigma[k] += 1e-6 * np.eye(D)  # стабилизация
    pi = Nk / N
    if it in (0, 1, 5, 10, 29):
        print(f"626-628 EM iter={it:>2}: log-lik={ll:.2f}, pi={pi.round(2)}")
    if abs(ll - prev_ll) < 1e-4:
        break
    prev_ll = ll

# Hard-assign по argmax responsibilities
labels = gamma.argmax(axis=1)
print(f"625 GMM итог: размеры кластеров={np.bincount(labels).tolist()}")
print(f"     mu[0]={mu[0].round(2)}, mu[1]={mu[1].round(2)}, mu[2]={mu[2].round(2)}")
