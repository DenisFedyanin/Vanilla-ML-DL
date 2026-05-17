"""Раздел 41 — Предобученные LM. Файл 2: использование предобученной модели.

Концепция: feature extraction.
Замораживаем все веса бэкбона, обучаем только classifier head на его выходах.
Быстро, мало данных, но качество может уступать full fine-tune.

Концепция: full fine-tuning.
Разморозим всё, обучаем end-to-end с маленьким lr (1e-5..5e-5).
Часто лучшее качество, но риск перегореть веса.

Концепция: discriminative LR / layer-wise LR decay.
Нижние слои — меньший lr, верхние — больший. Сохраняет общее знание, адаптирует верхушку.

Концепция: classifier head.
Над [CLS]-векторами (или mean-pool токенов) ставится Linear -> softmax.

Концепция: ELECTRA replaced-token detection.
Эффективнее MLM: модель учится по всем токенам, а не только 15% маскированных.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(0)

# === Имитируем предобученные эмбеддинги предложений ===
N, d = 60, 16
y = rng.integers(0, 2, size=N)
# Класс 0 — векторы вокруг (-1,...), класс 1 — около (+1,...)
mean0 = np.full(d, -0.5)
mean1 = np.full(d, +0.5)
X_emb = np.where(y[:, None] == 0, mean0, mean1) + rng.standard_normal((N, d)) * 0.3
print(f"Эмбеддинги{X_emb.shape}, метки{y.shape}")

# === Feature extraction: только LR на эмбеддингах ===
clf = LogisticRegression(max_iter=500).fit(X_emb, y)
print(f"\nFeature extraction (LR head): train acc = {clf.score(X_emb, y):.3f}")

# === Fine-tuning имитация: меняем эмбеддинги (1 шаг градиентного спуска) ===
# В реальной модели backbone обновляется; тут — упростим до сдвига эмбеддингов
X_ft = X_emb.copy()
for _ in range(5):
    logits = X_ft @ clf.coef_.T + clf.intercept_
    p = 1 / (1 + np.exp(-logits))
    grad = (p.flatten() - y)[:, None] * clf.coef_
    X_ft -= 0.01 * grad             # маленький lr — характерно для fine-tune
clf2 = LogisticRegression(max_iter=500).fit(X_ft, y)
print(f"После 'fine-tune' эмбеддингов: train acc = {clf2.score(X_ft, y):.3f}")

# === Discriminative LR concept ===
print("\n=== Layer-wise LR decay (концепция) ===")
n_layers = 12
base_lr = 1e-5
decay = 0.9
for L in [0, 5, 11]:
    lr_L = base_lr * (decay ** (n_layers - 1 - L))
    print(f"  слой {L}: lr = {lr_L:.2e}")
print("Нижние слои (универсальные признаки) обучаются медленнее, верхние быстрее.")

# === ELECTRA ===
print("\n=== ELECTRA replaced-token detection ===")
T = 10
seq = rng.integers(0, 100, size=T)
replaced = seq.copy()
positions = rng.uniform(size=T) < 0.2
replaced[positions] = rng.integers(0, 100, size=positions.sum())
labels = (replaced != seq).astype(int)
print(f"orig    : {seq.tolist()}")
print(f"replaced: {replaced.tolist()}")
print(f"labels  : {labels.tolist()} (1=заменён)")
print("Дискриминатор бинарно классифицирует КАЖДЫЙ токен -> в 7x эффективнее MLM на токен.")
