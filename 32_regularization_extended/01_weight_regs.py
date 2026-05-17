"""Раздел 32 — Регуляризация (расширенно). Файл 1: на весах.

Концепция 359: L1.
R = lambda * sum |w|. Делает веса разреженными (часть = 0). Эквивалент Лапласовского приора.

Концепция 360: L2 (Ridge / weight decay).
R = lambda * sum w^2. Сжимает все веса равномерно к нулю.
Эквивалент Гауссовского приора.

Концепция 361: ElasticNet.
R = lambda * (alpha*||w||_1 + (1-alpha)*||w||_2^2). Гибрид L1 и L2.

Концепция 362: MaxNorm.
Ограничение: ||w_row|| <= c. После шага градиента нормируем строки.
Не даёт весам "взорваться".

Концепция 363: Orthogonal regularization.
R = ||W^T W - I||_F^2. Стимулирует ортогональность W
(сохраняет норму активаций, спектр близкий к единице).

Концепция 364: Spectral norm (power iteration).
sigma_1(W) — максимальное сингулярное значение. Считается итерациями
u <- W v / ||W v||, v <- W^T u / ||W^T u||. Используется в Spectral Normalization.
"""
import numpy as np

rng = np.random.default_rng(0)
w = rng.standard_normal((5, 4))

# === 359: L1 ===
l1 = np.sum(np.abs(w))
print(f"359 L1(w) = {l1:.4f}")

# === 360: L2 ===
l2 = np.sum(w ** 2)
print(f"360 L2(w) = {l2:.4f}")

# === 361: ElasticNet ===
alpha = 0.5
en = alpha * np.sum(np.abs(w)) + (1 - alpha) * np.sum(w ** 2)
print(f"361 ElasticNet(alpha=0.5) = {en:.4f}")

# === 362: MaxNorm constraint (clip rows) ===
def max_norm(W, max_c):
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    scale = np.minimum(1.0, max_c / (norms + 1e-9))
    return W * scale

w_clipped = max_norm(w, max_c=1.0)
print(f"362 MaxNorm: до norms={np.linalg.norm(w, axis=1).round(3).tolist()}")
print(f"           после={np.linalg.norm(w_clipped, axis=1).round(3).tolist()}  (все <= 1.0)")

# === 363: Orthogonal regularization ===
ortho_reg = np.linalg.norm(w.T @ w - np.eye(4)) ** 2
print(f"363 Orthogonal reg ||W^T W - I||^2 = {ortho_reg:.4f}")

# Применим QR-ортогонализацию и снова посмотрим
Q, _ = np.linalg.qr(w)
ortho_reg_q = np.linalg.norm(Q.T @ Q - np.eye(4)) ** 2
print(f"    После QR-ортогонализации = {ortho_reg_q:.6f} (≈0)")

# === 364: Spectral norm via power iteration ===
def spectral_norm(W, n_iter=20):
    u = rng.standard_normal(W.shape[0]); u /= np.linalg.norm(u)
    for _ in range(n_iter):
        v = W.T @ u; v /= (np.linalg.norm(v) + 1e-9)
        u = W @ v;  u /= (np.linalg.norm(u) + 1e-9)
    return float(u @ W @ v)

sigma1_pi = spectral_norm(w)
sigma1_svd = np.linalg.svd(w, compute_uv=False)[0]
print(f"364 Spectral norm: power iter = {sigma1_pi:.4f}, точн. (SVD) = {sigma1_svd:.4f}")
