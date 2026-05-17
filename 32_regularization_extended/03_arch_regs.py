"""Раздел 32 — Регуляризация (расширенно). Файл 3: архитектурная.

Концепция 370: Stochastic depth.
С вероятностью p_drop пропускаем целый residual-блок: x_{l+1} = x_l.
Регуляризирует и удешевляет обучение глубоких сетей (ResNet, ViT).

Концепция 371: DropConnect.
Зануляем не активации, а отдельные ВЕСА матрицы. Более общая форма Dropout.

Концепция 372: DropBlock.
Зануляем не отдельные пиксели, а ПРЯМОУГОЛЬНЫЕ блоки на feature-map.
Сильнее регуляризует свёрточные сети, чем обычный Dropout.

Концепция 373: Zoneout (RNN dropout).
С вероятностью p сохраняем предыдущий h_{t-1} вместо вычисленного h_t.
Аналог dropout для RNN, сохраняющий информацию во времени.

Концепция 374: R-Drop (концепт).
Два прохода с разным dropout-маской -> два предсказания p1, p2.
Добавляем штраф KL(p1||p2) + KL(p2||p1) — заставляем модель быть согласованной.

Концепция 375: SAM — Sharpness-Aware Minimization (концепт).
Минимизируем max_{||eps||<=rho} L(w+eps), а не просто L(w).
Шаг: 1) идём вверх по grad на rho, 2) считаем grad там, 3) шаг по нему из w.
Делает минимум "плоским" — лучше обобщает.
"""
import numpy as np

rng = np.random.default_rng(0)

# === 370: Stochastic depth ===
def residual_block(x, F, p_drop, rng, train=True):
    if train and rng.uniform() < p_drop:
        return x                                  # дропнули блок
    return x + F(x)

def F(x): return 0.1 * x          # учебный residual
x = np.array([1.0, 1.0])
drops = sum(1 for _ in range(1000) if residual_block(x, F, 0.3, rng).tolist() == x.tolist())
print(f"370 Stochastic depth p=0.3: блок дропнули {drops}/1000 раз (~30%)")

# === 371: DropConnect ===
def drop_connect(W, p, rng, train=True):
    if not train or p == 0:
        return W
    mask = (rng.uniform(size=W.shape) > p).astype(float)
    return W * mask / (1 - p)

W = rng.standard_normal((5, 5))
W_dc = drop_connect(W, 0.5, rng)
print(f"371 DropConnect p=0.5: занулено весов = {(W_dc == 0).sum()}/{W.size}")

# === 372: DropBlock ===
def drop_block(x, block_size, p, rng, train=True):
    # x: (H, W)
    if not train or p == 0:
        return x
    H, W = x.shape
    mask = np.ones_like(x)
    # с вероятностью p ставим "центр" блока в маленьком пуле центров
    gamma = p / (block_size ** 2)
    centers = rng.uniform(size=(H, W)) < gamma
    cs, ys = np.where(centers)
    for i, j in zip(cs, ys):
        h0 = max(0, i - block_size // 2); h1 = min(H, i + block_size // 2 + 1)
        w0 = max(0, j - block_size // 2); w1 = min(W, j + block_size // 2 + 1)
        mask[h0:h1, w0:w1] = 0
    scale = mask.size / (mask.sum() + 1e-9)
    return x * mask * scale

img = np.ones((10, 10))
img_db = drop_block(img, block_size=3, p=0.2, rng=rng)
print(f"372 DropBlock: занулено пикселей = {(img_db == 0).sum()}/{img.size}, остальные масштабированы")

# === 373: Zoneout ===
def zoneout(h_prev, h_new, p, rng, train=True):
    if not train or p == 0:
        return h_new
    mask = (rng.uniform(size=h_new.shape) < p).astype(float)
    return mask * h_prev + (1 - mask) * h_new

h_prev = np.zeros(5); h_new = np.ones(5)
h = zoneout(h_prev, h_new, p=0.3, rng=rng)
print(f"373 Zoneout p=0.3: hidden = {h.tolist()} (часть = 0 от прошлого шага)")

# === 374: R-Drop (концепт) ===
def kl(p, q): return (p * (np.log(p + 1e-9) - np.log(q + 1e-9))).sum()
p1 = np.array([0.7, 0.2, 0.1])
p2 = np.array([0.65, 0.25, 0.1])     # два прохода с разным dropout
rdrop = 0.5 * (kl(p1, p2) + kl(p2, p1))
print(f"374 R-Drop KL-штраф = {rdrop:.4f}")

# === 375: SAM (один шаг) ===
def L(w): return 0.5 * np.sum(w ** 2)
def grad_L(w): return w
w = np.array([1.0, 2.0])
g = grad_L(w)
rho = 0.05
eps = rho * g / (np.linalg.norm(g) + 1e-9)
g_perturbed = grad_L(w + eps)
w_new = w - 0.1 * g_perturbed
print(f"375 SAM(rho={rho}): w={w.tolist()} -> w_new={w_new.round(4).tolist()}")
print("    Градиент берём в 'наихудшей' точке шара радиуса rho.")
