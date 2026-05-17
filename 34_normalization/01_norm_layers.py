"""Раздел 34 — Нормализация. Файл 1: основные слои.

Концепция 390: BatchNorm.
Нормируем по батчу: (x - mu_batch) / sqrt(var_batch + eps) -> gamma * + beta.
Train: статистики с текущего батча. Eval: накопленные running mean/var.
Стабилизирует обучение, ускоряет сходимость.

Концепция 391: LayerNorm.
Нормируем по ПРИЗНАКАМ внутри ОДНОГО объекта (по последней оси).
Не зависит от батча — стандарт в трансформерах и RNN.

Концепция 392: InstanceNorm.
Для (B, C, H, W): нормируем КАЖДУЮ карту (B,C) независимо по H,W.
Используется в стилевом переносе, GAN-ах.

Концепция 393: GroupNorm.
Каналы делим на G групп, нормируем внутри каждой группы.
При G=C это InstanceNorm; при G=1 — LayerNorm. Не зависит от батча,
хорош на маленьких батчах.
"""
import numpy as np

rng = np.random.default_rng(0)
eps = 1e-5

# === 390: BatchNorm ===
def batch_norm(x, gamma, beta, running_mean=None, running_var=None,
               train=True, momentum=0.1):
    if train:
        mu = x.mean(axis=0)
        var = x.var(axis=0)
        if running_mean is not None:
            running_mean[:] = (1 - momentum) * running_mean + momentum * mu
            running_var[:] = (1 - momentum) * running_var + momentum * var
    else:
        mu, var = running_mean, running_var
    xn = (x - mu) / np.sqrt(var + eps)
    return gamma * xn + beta

B, D = 32, 4
x = rng.standard_normal((B, D)) * 2 + 1   # mean=1, std=2
gamma, beta = np.ones(D), np.zeros(D)
rm, rv = np.zeros(D), np.ones(D)
y_train = batch_norm(x, gamma, beta, rm, rv, train=True)
print(f"390 BatchNorm train: y mean≈{y_train.mean(axis=0).round(3).tolist()} (~0)")
print(f"    y std≈{y_train.std(axis=0).round(3).tolist()} (~1)")
y_eval = batch_norm(x, gamma, beta, rm, rv, train=False)
print(f"    Eval использует running stats: y mean={y_eval.mean():.3f}")

# === 391: LayerNorm ===
def layer_norm(x, gamma, beta):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta

y_ln = layer_norm(x, gamma, beta)
print(f"391 LayerNorm: по каждому объекту mean≈{y_ln.mean(axis=1)[:3].round(3).tolist()} (~0)")

# === 392: InstanceNorm (для изображений) ===
def instance_norm(x, gamma, beta):
    # x: (B, C, H, W)
    mu = x.mean(axis=(2, 3), keepdims=True)
    var = x.var(axis=(2, 3), keepdims=True)
    g = gamma.reshape(1, -1, 1, 1); b = beta.reshape(1, -1, 1, 1)
    return g * (x - mu) / np.sqrt(var + eps) + b

img = rng.standard_normal((2, 3, 4, 4)) * 2
y_in = instance_norm(img, np.ones(3), np.zeros(3))
print(f"392 InstanceNorm: каждый (B,C)-канал mean≈0: {y_in.mean(axis=(2, 3)).round(3).tolist()}")

# === 393: GroupNorm ===
def group_norm(x, gamma, beta, G):
    B, C, H, W = x.shape
    x_g = x.reshape(B, G, C // G, H, W)
    mu = x_g.mean(axis=(2, 3, 4), keepdims=True)
    var = x_g.var(axis=(2, 3, 4), keepdims=True)
    x_g = (x_g - mu) / np.sqrt(var + eps)
    x = x_g.reshape(B, C, H, W)
    return gamma.reshape(1, -1, 1, 1) * x + beta.reshape(1, -1, 1, 1)

img2 = rng.standard_normal((2, 6, 4, 4)) * 2
y_gn = group_norm(img2, np.ones(6), np.zeros(6), G=2)   # 2 группы по 3 канала
print(f"393 GroupNorm G=2: общий mean={y_gn.mean():.3f} (~0), std={y_gn.std():.3f} (~1)")
print("   G=1 -> LayerNorm; G=C -> InstanceNorm.")
