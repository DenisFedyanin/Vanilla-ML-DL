"""Раздел 47 — Современные генеративные модели.

Файл 2: нормирующие потоки (Normalizing Flows).

Концепция: Change of variables.
Если z = f^{-1}(x) и f обратима, то p_X(x) = p_Z(f^{-1}(x)) * |det J_{f^{-1}}|.
Хотим f, у которой и f, и f^{-1}, и det Якобиана дёшево считаются.

Концепция: RealNVP coupling layer.
Делим вход на две части x_a, x_b. Преобразование:
y_a = x_a; y_b = x_b * exp(s(x_a)) + t(x_a).
Якобиан треугольный, det = exp(sum s). Обратимо: x_b = (y_b - t(y_a)) * exp(-s(y_a)).

Концепция: Glow 1x1 invertible conv.
Между coupling-слоями делается обратимая 1x1 conv (LU-параметризация). Смешивает
каналы; det через определитель LU-матриц.

Концепция: Log-likelihood обучения.
log p_X(x) = log p_Z(z) + log |det J|. Минимизируем -log p_X(x). И f, и
log|det J| дифференцируемы — обычный backprop.

Концепция: Sampling.
z ~ p_Z (стандартное Gaussian), x = f(z). Один forward — параллельно для всего
батча, в отличие от AR.
"""
import numpy as np

rng = np.random.default_rng(0)


# RealNVP coupling: 2D вход, разделяем на (x0, x1)
# s(x0) и t(x0) — простые линейные функции
def coupling_fwd(x, s_w, s_b, t_w, t_b):
    x0, x1 = x[:, 0], x[:, 1]
    s = s_w * x0 + s_b
    t = t_w * x0 + t_b
    y0 = x0
    y1 = x1 * np.exp(s) + t
    log_det = s  # для каждого примера
    return np.stack([y0, y1], axis=1), log_det


def coupling_inv(y, s_w, s_b, t_w, t_b):
    y0, y1 = y[:, 0], y[:, 1]
    s = s_w * y0 + s_b
    t = t_w * y0 + t_b
    x0 = y0
    x1 = (y1 - t) * np.exp(-s)
    return np.stack([x0, x1], axis=1)


# Игрушечные данные: спираль/диагональ
n = 500
x_data = rng.normal(size=(n, 2))
x_data[:, 1] = 2 * x_data[:, 0] + 0.3 * rng.normal(size=n)

# Обучение: max log p(x) под N(0,I) prior
s_w, s_b, t_w, t_b = 0.0, 0.0, 0.0, 0.0
lr = 0.01
for step in range(300):
    z, log_det = coupling_fwd(x_data, s_w, s_b, t_w, t_b)
    # log p(z) для N(0, I) (2D)
    log_pz = -0.5 * (z ** 2).sum(axis=1) - np.log(2 * np.pi)
    log_px = log_pz + log_det
    nll = -log_px.mean()
    # Численный градиент
    eps = 1e-4
    g = []
    for name in ("s_w", "s_b", "t_w", "t_b"):
        loc = {"s_w": s_w, "s_b": s_b, "t_w": t_w, "t_b": t_b}
        loc[name] += eps
        z2, ld2 = coupling_fwd(x_data, loc["s_w"], loc["s_b"], loc["t_w"], loc["t_b"])
        nll2 = -(-0.5 * (z2 ** 2).sum(axis=1) - np.log(2 * np.pi) + ld2).mean()
        g.append((nll2 - nll) / eps)
    s_w -= lr * g[0]
    s_b -= lr * g[1]
    t_w -= lr * g[2]
    t_b -= lr * g[3]

print(f"Выучили s_w={s_w:.2f}, s_b={s_b:.2f}, t_w={t_w:.2f}, t_b={t_b:.2f}")
print(f"Финальный NLL: {nll:.3f}")

# Sampling: z ~ N(0,I) -> x = f^{-1}? Нет — f отображает x->z; нужно обратное
# В нашей нотации coupling_inv делает y -> x. Семплим z, считаем "обратное" к forward:
z_samp = rng.normal(size=(5, 2))
# В RealNVP forward x->z, для семплинга нужно решить уравнения; в нашем coupling
# y = forward(x); чтобы из z получить x, применяем coupling_inv(z) -> это вернёт x
x_samp = coupling_inv(z_samp, s_w, s_b, t_w, t_b)
print("Сгенерированные точки (5 шт.):")
print(x_samp.round(2))

# Glow 1x1 conv: концепт — обратимая матрица W (2x2), x_new = W @ x
W = np.array([[1.0, 0.2], [-0.1, 0.9]])
det_W = np.linalg.det(W)
log_det_W = np.log(abs(det_W))
print(f"\n1x1 conv det={det_W:.3f}, log|det|={log_det_W:.3f}")
