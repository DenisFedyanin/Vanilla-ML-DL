"""Раздел 25 — Поиск аномалий. Файл 3: реконструкция (autoencoder).

Концепция 688: Autoencoder для аномалий — идея.
Учим сеть x -> z -> x_hat реконструировать «нормальные» данные. На аномалиях
ошибка реконструкции велика, потому что сеть никогда таких не видела.

Концепция 689: Bottleneck.
Размер z (латентного слоя) меньше размера входа — сеть вынуждена «сжимать»
типичные паттерны. Без bottleneck могла бы выучить тождественную функцию.

Концепция 690: Скор аномальности.
score = ||x - x_hat||^2. Аномалия — точки с большим score (например, выше
95-го перцентиля train-ошибок).

Реализация: ручной MLP-autoencoder на numpy, обучаем градиентным спуском.
"""
import numpy as np

rng = np.random.default_rng(0)

# === нормальные данные на 2D-многообразии (плоскость) в 5D ===
N, D, K = 200, 5, 2
W_gen = rng.normal(size=(K, D))
Z = rng.normal(size=(N, K))
X_norm = Z @ W_gen
X_norm += 0.05 * rng.normal(size=X_norm.shape)
# выбросы — точки не на многообразии
X_out = rng.uniform(-3, 3, size=(20, D))
X = np.vstack([X_norm, X_out])
y_anom = np.concatenate([np.zeros(N), np.ones(20)])

# === 688-689: ручной линейный autoencoder D -> 2 -> D ===
hidden = 2
W1 = rng.normal(scale=0.3, size=(D, hidden))
W2 = rng.normal(scale=0.3, size=(hidden, D))
lr = 0.01

# Учимся ТОЛЬКО на нормальных
for epoch in range(300):
    Zh = X_norm @ W1
    Xh = Zh @ W2
    err = Xh - X_norm
    # градиенты MSE по линейному auto-enc
    gW2 = Zh.T @ err / N
    gW1 = X_norm.T @ (err @ W2.T) / N
    W1 -= lr * gW1
    W2 -= lr * gW2

train_err = ((X_norm @ W1 @ W2 - X_norm) ** 2).sum(axis=1)
all_err = ((X @ W1 @ W2 - X) ** 2).sum(axis=1)
thr = np.percentile(train_err, 95)
pred = (all_err > thr).astype(int)

tp = ((pred == 1) & (y_anom == 1)).sum()
fp = ((pred == 1) & (y_anom == 0)).sum()
print(f"688-689 Autoencoder MSE на train={train_err.mean():.4f}, "
      f"порог (95-й перцентиль)={thr:.4f}")
print(f"690 На всех данных: TP={tp}/20 аномалий, FP={fp}/{N} нормальных")
print(f"     средняя reconstruction-error: норм={all_err[:N].mean():.3f}, "
      f"выбросы={all_err[N:].mean():.3f}")
