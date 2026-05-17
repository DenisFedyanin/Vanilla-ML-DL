"""Раздел 68 — Kernel methods.

Файл 3: Приближения ядер (для масштабируемости).

Концепция: Зачем приближать?
Gram-матрица K имеет размер n x n, обращение O(n^3), память O(n^2). Не работает
при n > 10^4 - 10^5. Нужны приближения с явным признаковым представлением phi_hat(x),
чтобы K(x, y) ~ phi_hat(x)^T phi_hat(y) и работать в линейных моделях с O(n*m).

Концепция: Nyström approximation.
Сэмплируем m << n опорных точек (landmarks). Считаем W = K(landmarks, landmarks) (m x m)
и C = K(all, landmarks) (n x m). Тогда K ~ C W^+ C^T, где W^+ — псевдообратное.
Эквивалентно: phi_hat(x) = W^{-1/2} K(x, landmarks).

Концепция: Random Fourier Features (RFF) для RBF.
Теорема Бохнера: shift-invariant kernel = преобразование Фурье от вероятностной меры.
Для RBF (gamma=1/2sigma^2): сэмплируем w_i ~ N(0, 2*gamma * I), b_i ~ U[0, 2pi].
Признак: z(x) = sqrt(2/D) * cos(w_i^T x + b_i), i=1..D. Тогда z(x)^T z(y) ~ K_RBF(x, y).

Концепция: Compare approximation vs true.
Сравниваем Frobenius-разность Gram-матриц true K и approx phi_hat phi_hat^T.
Чем больше m / D, тем точнее.
"""
import numpy as np

rng = np.random.default_rng(0)

def rbf_kernel(X, Y, gamma=0.5):
    sq = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    return np.exp(-gamma * sq)

n, d = 200, 4
X = rng.normal(size=(n, d))
gamma = 0.5
K_true = rbf_kernel(X, X, gamma)
print(f"True Gram: shape={K_true.shape}, trace={np.trace(K_true):.1f}")

# === Nystrom: m landmarks ===
def nystrom(X, gamma, m, rng):
    idx = rng.choice(len(X), size=m, replace=False)
    W = rbf_kernel(X[idx], X[idx], gamma)        # m x m
    C = rbf_kernel(X, X[idx], gamma)             # n x m
    # эффективные признаки: phi = C @ W^{-1/2}
    eigvals, eigvecs = np.linalg.eigh(W)
    eigvals = np.maximum(eigvals, 1e-10)
    W_inv_half = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    phi = C @ W_inv_half
    return phi, idx

for m in [20, 50, 100]:
    phi, _ = nystrom(X, gamma, m, rng)
    K_approx = phi @ phi.T
    err = np.linalg.norm(K_true - K_approx, "fro") / np.linalg.norm(K_true, "fro")
    print(f"Nystrom m={m:3d}: rel Frobenius error = {err:.4f}")

# === Random Fourier Features ===
def rff(X, gamma, D, rng):
    d = X.shape[1]
    W = rng.normal(scale=np.sqrt(2 * gamma), size=(d, D))
    b = rng.uniform(0, 2 * np.pi, size=D)
    return np.sqrt(2.0 / D) * np.cos(X @ W + b)

for D in [50, 200, 1000]:
    phi = rff(X, gamma, D, np.random.default_rng(D))
    K_approx = phi @ phi.T
    err = np.linalg.norm(K_true - K_approx, "fro") / np.linalg.norm(K_true, "fro")
    print(f"RFF     D={D:4d}: rel Frobenius error = {err:.4f}")

# === пример скорости ===
# линейная ridge на RFF-фичах вместо kernel ridge:
y = np.sin(X[:, 0]) + 0.1 * rng.normal(size=n)
phi = rff(X, gamma, D=500, rng=np.random.default_rng(1))
# w = (Phi^T Phi + lam I)^-1 Phi^T y
lam = 1.0
w = np.linalg.solve(phi.T @ phi + lam * np.eye(phi.shape[1]), phi.T @ y)
y_pred = phi @ w
mse_rff = ((y_pred - y) ** 2).mean()

# для сравнения — точный KRR
alpha = np.linalg.solve(K_true + lam * np.eye(n), y)
y_pred_kr = K_true @ alpha
mse_kr = ((y_pred_kr - y) ** 2).mean()
print(f"\nMSE: KRR exact = {mse_kr:.4f}, Ridge on RFF features = {mse_rff:.4f}")
print("RFF делает ядро 'линейным' — можно использовать любые быстрые линейные методы.")
