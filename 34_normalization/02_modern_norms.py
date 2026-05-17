"""Раздел 34 — Нормализация. Файл 2: современные.

Концепция 394: RMSNorm.
y = x / RMS(x) * gamma, где RMS(x) = sqrt(mean(x^2)+eps).
Без вычитания среднего — дешевле LayerNorm. Стандарт в LLaMA, T5.

Концепция 395: WeightNorm.
W = (g / ||v||) * v. Параметризуем веса как направление v и норму g.
Ускоряет обучение, де-связывает направление и масштаб.

Концепция 396: Spectral Normalization (power iteration).
W_norm = W / sigma_1(W), где sigma_1 — наибольшее сингулярное значение.
Гарантирует ||W||_2 = 1. Стабилизирует GAN-дискриминаторы.

Концепция 397: ActNorm.
Инициализируется по ПЕРВОМУ батчу: подбираем scale/bias так,
чтобы выход имел mean=0, std=1. Далее параметры обучаются. Стандарт в Glow.

Концепция 398: Batch Renorm (концепт).
Чинит проблему BatchNorm на маленьких батчах: использует "rescaling"
r = sigma_batch/sigma_running, d = (mu_batch-mu_running)/sigma_running,
прижатые к окрестности 1 и 0. На train ведёт себя как BN, но с поправкой к running stats.

Концепция 399: Filter Response Norm (FRN, концепт).
y = x / sqrt(mean(x^2)+eps) (по H,W). Не зависит ни от батча, ни от каналов.
Применяется вместе с TLU (Thresholded Linear Unit).
"""
import numpy as np

rng = np.random.default_rng(0)
eps = 1e-5

# === 394: RMSNorm ===
def rms_norm(x, gamma):
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return gamma * x / rms

x = rng.standard_normal((4, 6)) * 2
y = rms_norm(x, np.ones(6))
print(f"394 RMSNorm: каждая строка RMS≈{np.sqrt((y ** 2).mean(axis=1)).round(3).tolist()} (~1)")

# === 395: WeightNorm ===
def weight_norm(v, g):
    return g * v / (np.linalg.norm(v) + 1e-9)

v = rng.standard_normal((4, 4))
g = 2.0
W = weight_norm(v, g)
print(f"395 WeightNorm: ||W||_F = {np.linalg.norm(W):.3f}, целевое = {g}")

# === 396: Spectral Normalization ===
def spectral_norm_pi(W, n_iter=20):
    u = rng.standard_normal(W.shape[0]); u /= np.linalg.norm(u)
    for _ in range(n_iter):
        v = W.T @ u; v /= (np.linalg.norm(v) + 1e-9)
        u = W @ v;  u /= (np.linalg.norm(u) + 1e-9)
    sigma = float(u @ W @ v)
    return W / sigma, sigma

W_orig = rng.standard_normal((5, 5)) * 3
W_sn, sigma = spectral_norm_pi(W_orig)
print(f"396 SpectralNorm: было sigma_max={sigma:.3f}, стало sigma_max≈{np.linalg.svd(W_sn, compute_uv=False)[0]:.3f}")

# === 397: ActNorm (data-dependent init) ===
class ActNorm:
    def __init__(self, dim):
        self.scale = None; self.bias = None; self.dim = dim
    def initialize(self, x):
        mu = x.mean(axis=0)
        std = x.std(axis=0)
        self.bias = -mu / (std + eps)
        self.scale = 1 / (std + eps)
    def __call__(self, x):
        if self.scale is None:
            self.initialize(x)
        return self.scale * x + self.bias

an = ActNorm(dim=4)
batch1 = rng.standard_normal((32, 4)) * 5 + 3
y1 = an(batch1)
print(f"397 ActNorm: после init mean={y1.mean(axis=0).round(3).tolist()}, std={y1.std(axis=0).round(3).tolist()}")

# === 398: Batch Renorm (концепт) ===
def batch_renorm(x, running_mean, running_var, r_max=3.0, d_max=5.0):
    mu_b = x.mean(axis=0); var_b = x.var(axis=0)
    sigma_b = np.sqrt(var_b + eps); sigma_r = np.sqrt(running_var + eps)
    r = np.clip(sigma_b / sigma_r, 1 / r_max, r_max)
    d = np.clip((mu_b - running_mean) / sigma_r, -d_max, d_max)
    return ((x - mu_b) / sigma_b) * r + d

rm, rv = np.zeros(4), np.ones(4)
y_br = batch_renorm(x[:, :4], rm, rv)
print(f"398 BatchRenorm: mean={y_br.mean(axis=0).round(3).tolist()} (приближается к running)")

# === 399: FRN (концепт) ===
def frn(x, tau):
    # x: (B, C, H, W); нормируем по H,W
    rms = np.sqrt((x ** 2).mean(axis=(2, 3), keepdims=True) + eps)
    return np.maximum(x / rms, tau)            # с TLU

img = rng.standard_normal((2, 3, 4, 4))
y_frn = frn(img, tau=-1.0)
print(f"399 FRN+TLU: out{y_frn.shape}, min={y_frn.min():.3f}, max={y_frn.max():.3f}")
