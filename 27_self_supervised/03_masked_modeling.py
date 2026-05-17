"""Раздел 27 — Self-supervised learning. Файл 3: masked modeling.

Концепция 712: Masked Language Modeling (MLM).
Часть токенов скрывается специальным [MASK]; модель учится предсказать их
по контексту. Основа BERT-подобных моделей. Учим двунаправленные представления
без авторегрессии.

Концепция 713: Masked Autoencoder (MAE).
То же в зрении: входной картинке маскируют ~75% патчей, encoder видит только
видимые, decoder восстанавливает пиксели маскированных патчей.
Высокая доля маскирования заставляет модель учить семантику, а не локальные
дополнения.

Концепция 714: Цель: восстановить скрытое по видимому.
loss = MSE(predicted_hidden, true_hidden) — для MAE,
loss = CE(predicted_token, true_token)  — для MLM.

Реализация: линейная модель, которая по «видимым» координатам вектора
предсказывает «маскированные».
"""
import numpy as np

rng = np.random.default_rng(0)

# === 712-713: данные — 200 векторов размерности 10, генеренные с зависимостями ===
N, D = 200, 10
A = rng.normal(size=(D, D)) * 0.3
X = rng.normal(size=(N, D)) @ A  # коррелированные признаки

# Случайно маскируем 30% координат для каждого вектора
mask_ratio = 0.3
mask = rng.uniform(size=X.shape) < mask_ratio
visible = X.copy()
visible[mask] = 0.0  # «спрятали» — обнулили

# === 714: линейная модель: full = W @ visible (D x D), учим на MSE ===
# Решение через нормальное уравнение по выборке
# (берём только примеры, где хотя бы что-то замаскировано — почти все)
W = np.linalg.lstsq(visible, X, rcond=None)[0]  # (D, D)

pred = visible @ W
mse_total = ((pred - X) ** 2).mean()
mse_masked = ((pred[mask] - X[mask]) ** 2).mean()
mse_visible = ((pred[~mask] - X[~mask]) ** 2).mean()
print(f"712-714 MAE-style: mask_ratio={mask_ratio:.0%}")
print(f"     MSE total={mse_total:.3f}, "
      f"MSE на маскированных={mse_masked:.3f}, "
      f"MSE на видимых={mse_visible:.3f}")

# Baseline: предсказывать ноль для маскированных
mse_zero = ((0 - X[mask]) ** 2).mean()
print(f"     Baseline (предсказать 0 для маскированных): MSE={mse_zero:.3f}")
print(f"     Улучшение MAE-модели: x{mse_zero / max(mse_masked, 1e-9):.1f}")

# === 712: MLM мини-демо ===
# Возьмём «слова» 0..4, последовательности длины 5, замаскируем одно — предскажем
# по соседям через простой подсчёт совместных частот (биграммы).
vocab = 5
seqs = rng.integers(0, vocab, size=(500, 5))
# биграммы (i, j)
bg = np.zeros((vocab, vocab))
for s in seqs:
    for a, b in zip(s[:-1], s[1:]):
        bg[a, b] += 1
bg /= bg.sum(axis=1, keepdims=True) + 1e-9
# для каждой последовательности маскируем средний токен и предсказываем по левому соседу
preds = bg[seqs[:, 1]].argmax(axis=1)
acc = (preds == seqs[:, 2]).mean()
print(f"712 MLM-toy: accuracy предсказания маскированного токена по соседу={acc:.3f} "
      f"(случайно бы было {1/vocab:.2f})")
