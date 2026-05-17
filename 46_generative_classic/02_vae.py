"""Раздел 46 — Классические генеративные модели.

Файл 2: VAE (Variational Autoencoder).

Концепция: VAE.
Encoder q(z|x) -> mu, log_var. Decoder p(x|z). Скрытое z ~ N(mu, sigma)
семплится при обучении (reparam trick). Учим максимизировать ELBO.

Концепция: ELBO (Evidence Lower BOund).
log p(x) >= E_q[log p(x|z)] - KL(q(z|x) || p(z)).
Первый член — реконструкция, второй — регуляризация (приближение q к prior N(0,I)).

Концепция: Reparameterization trick.
z = mu + sigma * eps, eps ~ N(0, I). Перенос стохастики на eps делает
mu, sigma дифференцируемыми -> можем учить backprop'ом.

Концепция: KL для N(mu, sigma) || N(0, I).
Аналитически: 0.5 * sum(mu^2 + sigma^2 - log(sigma^2) - 1) по координатам z.

Концепция: Decoder.
Из z -> x (или mean(x)). Можно учить как MSE (Gaussian likelihood) или
BCE (Bernoulli likelihood) для бинарных изображений.
"""
import numpy as np

rng = np.random.default_rng(0)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


# Игрушечные 4D данные, лежащие на 1D подпространстве с шумом
n = 200
t = rng.normal(size=n)
X = np.stack([t, 2 * t, -t, 0.5 * t], axis=1) + rng.normal(0, 0.1, size=(n, 4))

# VAE: encoder 4->h(8)->mu(1), log_var(1); decoder 1->h(8)->4
d, h, z_dim = 4, 8, 1
W1 = rng.normal(0, 0.5, size=(d, h))
W_mu = rng.normal(0, 0.5, size=(h, z_dim))
W_lv = rng.normal(0, 0.5, size=(h, z_dim))
W2 = rng.normal(0, 0.5, size=(z_dim, h))
W3 = rng.normal(0, 0.5, size=(h, d))
lr = 0.01

for step in range(400):
    # Forward
    H1 = np.tanh(X @ W1)
    mu = H1 @ W_mu
    log_var = H1 @ W_lv
    sigma = np.exp(0.5 * log_var)
    eps = rng.normal(size=mu.shape)
    z = mu + sigma * eps
    H2 = np.tanh(z @ W2)
    x_rec = H2 @ W3
    # ELBO loss = recon (MSE) + KL
    recon = ((x_rec - X) ** 2).mean()
    kl = 0.5 * np.mean(mu ** 2 + sigma ** 2 - log_var - 1)
    loss = recon + kl
    if step % 100 == 0:
        print(f"step {step}: recon={recon:.4f}, KL={kl:.4f}, loss={loss:.4f}")
    # Простой градиент (через конечные разности на одном шаге было бы дорого;
    # сделаем аналитически только для recon относительно decoder W3, W2,
    # остальное — простой "толчок" к лучшим mu/log_var).
    # Decoder backprop
    dx = 2 * (x_rec - X) / X.size
    dW3 = H2.T @ dx
    dH2 = dx @ W3.T
    dz = dH2 * (1 - H2 ** 2)
    dW2 = z.T @ dz
    # Encoder
    dmu = dz @ W2.T + mu / X.shape[0]
    dlv = dz @ W2.T * 0.5 * eps * sigma + 0.5 * (sigma ** 2 - 1) / X.shape[0]
    dH1_mu = dmu @ W_mu.T
    dH1_lv = dlv @ W_lv.T
    dH1 = (dH1_mu + dH1_lv) * (1 - H1 ** 2)
    dW_mu = H1.T @ dmu
    dW_lv = H1.T @ dlv
    dW1 = X.T @ dH1
    # обновления
    W1 -= lr * dW1
    W_mu -= lr * dW_mu
    W_lv -= lr * dW_lv
    W2 -= lr * dW2
    W3 -= lr * dW3

# Сгенерируем новые из prior z ~ N(0,1)
z_new = rng.normal(size=(5, z_dim))
H2 = np.tanh(z_new @ W2)
X_new = H2 @ W3
print("Сгенерированные образцы (5 шт., 4D):")
print(X_new.round(2))
print("Данные имели соотношения примерно (t, 2t, -t, 0.5t) — проверь, сохранились ли ratio")
