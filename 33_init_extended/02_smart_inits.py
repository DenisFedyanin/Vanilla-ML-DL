"""Раздел 33 — Инициализация (расширенно). Файл 2: умные схемы.

Идея: подобрать дисперсию весов так, чтобы дисперсия активаций не уплывала
по слоям (vanishing / exploding).

Концепция 382: Xavier (Glorot) Uniform.
W ~ U(-a, a), a = sqrt(6/(fan_in+fan_out)). Под tanh/sigmoid.

Концепция 383: Xavier Normal.
W ~ N(0, sqrt(2/(fan_in+fan_out))).

Концепция 384: He Uniform.
W ~ U(-a, a), a = sqrt(6/fan_in). Под ReLU (учитывает зануление половины).

Концепция 385: He Normal.
W ~ N(0, sqrt(2/fan_in)).

Концепция 386: LeCun.
W ~ N(0, sqrt(1/fan_in)) или U(...). Под SELU/линейные слои.

Концепция 387: Orthogonal init.
W = Q из QR-разложения случайной матрицы. Сохраняет норму, спектр = 1.
Хорошо для RNN, deep CNN.

Концепция 388: Identity init.
W = I (квадратная). Полезно для рекуррентных и residual-сетей —
старт с тождественного преобразования.

Концепция 389: Variance scaling.
Обобщённая семья: sigma^2 = scale / fan_X, где X ∈ {in, out, avg}.
Большинство известных инициализаций — её частные случаи.
"""
import numpy as np

rng = np.random.default_rng(0)
fan_in, fan_out = 512, 512
n_layers = 5
x = rng.standard_normal(fan_in)        # вход стандартизован

def relu(z): return np.maximum(0, z)
def tanh(z): return np.tanh(z)

def forward(x, scheme, activ, depth=n_layers):
    a = x
    for _ in range(depth):
        if scheme == "xavier_u":
            lim = np.sqrt(6 / (fan_in + fan_out))
            W = rng.uniform(-lim, lim, (fan_out, fan_in))
        elif scheme == "xavier_n":
            W = rng.normal(0, np.sqrt(2 / (fan_in + fan_out)), (fan_out, fan_in))
        elif scheme == "he_u":
            lim = np.sqrt(6 / fan_in)
            W = rng.uniform(-lim, lim, (fan_out, fan_in))
        elif scheme == "he_n":
            W = rng.normal(0, np.sqrt(2 / fan_in), (fan_out, fan_in))
        elif scheme == "lecun":
            W = rng.normal(0, np.sqrt(1 / fan_in), (fan_out, fan_in))
        elif scheme == "ortho":
            A = rng.standard_normal((fan_out, fan_in))
            Q, _ = np.linalg.qr(A)
            W = Q
        elif scheme == "identity":
            W = np.eye(fan_out, fan_in)
        elif scheme == "var_scaling":
            scale = 2.0
            W = rng.normal(0, np.sqrt(scale / fan_in), (fan_out, fan_in))
        a = activ(W @ a)
    return a

# Под ReLU
for name, scheme in [("382 Xavier U", "xavier_u"), ("383 Xavier N", "xavier_n"),
                     ("384 He U", "he_u"), ("385 He N", "he_n"),
                     ("386 LeCun", "lecun"), ("387 Orthogonal", "ortho"),
                     ("388 Identity", "identity"), ("389 VarianceScaling", "var_scaling")]:
    out = forward(x, scheme, relu)
    print(f"{name:25s}: std активаций после {n_layers} слоёв (ReLU) = {out.std():.3f}")

print("\nПод tanh (Xavier подходит лучше):")
for name, scheme in [("Xavier N", "xavier_n"), ("He N", "he_n")]:
    out = forward(x, scheme, tanh)
    print(f"  {name}: std = {out.std():.3f}")
