"""Раздел 26 — Semi-supervised. Файл 1: self-training / pseudo-labeling.

Концепция 691: Semi-supervised learning — постановка.
Есть мало размеченных (L) и много неразмеченных (U) данных. Хочется
использовать U, чтобы улучшить классификатор, обученный только на L.

Концепция 692: Self-training (pseudo-labeling).
1) обучаем модель на L; 2) предсказываем U; 3) самые уверенные предсказания
добавляем в L как «псевдо-метки»; 4) переобучаем; повторяем.

Концепция 693: Порог уверенности.
В качестве уверенности обычно берут max(predict_proba). Псевдо-метки
добавляем, только если уверенность выше порога (например, 0.9). Низкий порог —
шум, высокий — мало пользы.

Концепция 694: Риск дрейфа ошибок.
Если ранние псевдо-метки ошибочны, модель закрепляет ошибки. Поэтому
часто применяют curriculum: сначала только очень уверенные, потом
постепенно понижаем порог.
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(0)
X, y = make_classification(n_samples=400, n_features=10, n_informative=5,
                           random_state=0)
# === 691: разметим только 5% ===
n_lab = 20
idx = rng.permutation(len(X))
lab_idx = idx[:n_lab]
unlab_idx = idx[n_lab:]
mask_lab = np.zeros(len(X), dtype=bool)
mask_lab[lab_idx] = True

X_l, y_l = X[mask_lab], y[mask_lab]
X_u = X[~mask_lab]
y_u_true = y[~mask_lab]

# Базовая модель только на L
base = LogisticRegression(max_iter=500, random_state=0).fit(X_l, y_l)
acc_base = base.score(X_u, y_u_true)
print(f"691 Baseline (только {n_lab} меток): test-acc на U={acc_base:.3f}")

# === 692-694: self-training с порогом ===
X_train = X_l.copy()
y_train = y_l.copy()
remain_mask = np.ones(len(X_u), dtype=bool)
thr = 0.9
for it in range(5):
    clf = LogisticRegression(max_iter=500, random_state=0).fit(X_train, y_train)
    if not remain_mask.any():
        break
    probs = clf.predict_proba(X_u[remain_mask])
    conf = probs.max(axis=1)
    add_local = np.where(conf >= thr)[0]
    if len(add_local) == 0:
        thr -= 0.05  # curriculum: ослабляем порог
        continue
    remain_idx = np.where(remain_mask)[0][add_local]
    pseudo_y = clf.predict(X_u[remain_idx])
    X_train = np.vstack([X_train, X_u[remain_idx]])
    y_train = np.concatenate([y_train, pseudo_y])
    remain_mask[remain_idx] = False
    print(f"692-693 iter={it}: добавлено {len(remain_idx)} псевдо-меток "
          f"(thr={thr:.2f}), осталось U={remain_mask.sum()}")

final = LogisticRegression(max_iter=500, random_state=0).fit(X_train, y_train)
acc_final = final.score(X_u, y_u_true)
print(f"694 Итог self-training: acc на U={acc_final:.3f} (было {acc_base:.3f})")
