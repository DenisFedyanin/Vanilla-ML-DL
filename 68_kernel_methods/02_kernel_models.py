"""Раздел 68 — Kernel methods.

Файл 2: Kernel Ridge, Kernel PCA, Kernel K-means.

Концепция: Kernel Ridge Regression (KRR).
В признаковом пространстве: w = (Phi^T Phi + lambda I)^-1 Phi^T y.
Через двойственность (Representer theorem): f(x) = sum_i alpha_i K(x_i, x),
где alpha = (K + lambda I)^-1 y. Никогда не работаем с Phi напрямую.

Концепция: Kernel PCA.
Центрируем Gram-матрицу K: K_c = K - 1_n K - K 1_n + 1_n K 1_n (1_n = J/n).
Делаем EV-разложение K_c = sum lambda_i v_i v_i^T. Координаты проекции точки
x на k-ю компоненту: sum_i v_{i,k} / sqrt(lambda_k) * K(x_i, x).

Концепция: Kernel K-means.
Минимизируем sum_c sum_{i in c} ||phi(x_i) - mu_c||^2 в feature space.
||phi(x_i) - mu_c||^2 = K_ii - (2/|c|) sum_{j in c} K_ij + (1/|c|^2) sum_{j,k in c} K_jk.
Работает с нелинейно разделимыми кластерами (концентрические круги и т.п.).

Демо: KRR на синусе с RBF; Kernel PCA на 2 кольцах (становятся линейно
разделимыми в feature space).
"""
import numpy as np
from sklearn.datasets import make_circles

rng = np.random.default_rng(0)

def rbf(X, Y, gamma=0.5):
    sq = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    return np.exp(-gamma * sq)

# === Kernel Ridge Regression: y = sin(x) ===
X_tr = np.linspace(-3, 3, 30).reshape(-1, 1)
y_tr = np.sin(X_tr).ravel() + 0.1 * rng.normal(size=30)
lam = 0.1
gamma = 1.0
K = rbf(X_tr, X_tr, gamma)
alpha = np.linalg.solve(K + lam * np.eye(len(X_tr)), y_tr)

X_te = np.linspace(-3, 3, 100).reshape(-1, 1)
K_te = rbf(X_te, X_tr, gamma)
y_pred = K_te @ alpha
y_true = np.sin(X_te).ravel()
mse = ((y_pred - y_true) ** 2).mean()
print(f"Kernel Ridge Regression: MSE на тесте = {mse:.4f}")
print(f"  ||alpha||={np.linalg.norm(alpha):.2f}, gamma={gamma}, lambda={lam}")

# === Kernel PCA на 2 кольцах ===
X, y = make_circles(n_samples=100, factor=0.3, noise=0.05, random_state=0)
K = rbf(X, X, gamma=10.0)
n = len(X)
one_n = np.ones((n, n)) / n
K_c = K - one_n @ K - K @ one_n + one_n @ K @ one_n

eigvals, eigvecs = np.linalg.eigh(K_c)
eigvals = eigvals[::-1]
eigvecs = eigvecs[:, ::-1]
# проекция всех точек на 2 первые компоненты
alphas = eigvecs[:, :2] / np.sqrt(np.maximum(eigvals[:2], 1e-12))
proj = K_c @ alphas

# проверка: насколько лучше теперь разделимы классы по 1-й компоненте
mean0 = proj[y == 0, 0].mean()
mean1 = proj[y == 1, 0].mean()
sep_kpca = abs(mean0 - mean1) / (proj[:, 0].std() + 1e-12)

# линейный PCA для сравнения
Xc = X - X.mean(0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
proj_lin = Xc @ Vt[:2].T
sep_lin = abs(proj_lin[y == 0, 0].mean() - proj_lin[y == 1, 0].mean()) / (proj_lin[:, 0].std() + 1e-12)
print(f"Kernel PCA: separation на 1-й компоненте = {sep_kpca:.2f}")
print(f"Linear PCA: separation на 1-й компоненте = {sep_lin:.2f}")

# === Kernel K-means (2 кластера на тех же кольцах) ===
def kernel_kmeans(K, n_clusters=2, n_iter=20, rng=None):
    n = len(K)
    labels = rng.integers(0, n_clusters, size=n)
    diag = np.diag(K)
    for _ in range(n_iter):
        dists = np.zeros((n, n_clusters))
        for c in range(n_clusters):
            idx = np.where(labels == c)[0]
            if len(idx) == 0:
                dists[:, c] = np.inf
                continue
            t2 = K[:, idx].sum(1) / len(idx)
            t3 = K[np.ix_(idx, idx)].sum() / (len(idx) ** 2)
            dists[:, c] = diag - 2 * t2 + t3
        new_labels = dists.argmin(1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
    return labels

labels_kkm = kernel_kmeans(K, n_clusters=2, rng=rng)
# accuracy относительно истинных y (учитываем перестановку меток)
acc = max((labels_kkm == y).mean(), (labels_kkm != y).mean())
print(f"Kernel K-means: accuracy = {acc:.3f} (линейный k-means даёт ~0.5 на кольцах)")
