"""Раздел 18 — Семейство SVM.

Файл 1: hard margin vs soft margin и функции потерь.

Концепция 264: Hard-margin SVM (концепт).
Существует только при линейной разделимости. Максимизирует geometric margin
2/||w|| при условии y_i (w x_i + b) >= 1. На практике редко применим.

Концепция 265: Soft-margin SVM с параметром C.
Добавляем slack-переменные xi_i >= 0 и штраф C * sum xi_i. Малое C =>
большая допустимость ошибок (широкий margin). Большое C => почти hard-margin.

Концепция 266: Hinge vs squared hinge.
hinge: max(0, 1 - y * f(x)) — линейный штраф за нарушение margin.
squared hinge: max(0, 1 - y * f(x))^2 — квадратичный, гладкий, сильнее
наказывает большие нарушения.
"""
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC

rng = np.random.default_rng(0)

# 264: линейно разделимый набор -> hard-margin аналог (C очень большой)
X_sep, y_sep = make_blobs(n_samples=80, centers=[(-3, -3), (3, 3)], cluster_std=0.8, random_state=0)
hard = LinearSVC(C=1e4, loss="hinge", max_iter=10_000).fit(X_sep, y_sep)
print(f"264 hard-margin (C=1e4, разделимые): train acc={hard.score(X_sep, y_sep):.3f}")

# 265: soft-margin на пересекающихся данных, разный C
X_mix, y_mix = make_blobs(n_samples=200, centers=[(-1, -1), (1, 1)], cluster_std=1.5, random_state=0)
for C in (0.01, 1.0, 100.0):
    svm = LinearSVC(C=C, loss="hinge", max_iter=10_000).fit(X_mix, y_mix)
    print(f"265 soft-margin C={C}: acc={svm.score(X_mix, y_mix):.3f}, ||w||={np.linalg.norm(svm.coef_):.3f}")

# 266: hinge vs squared hinge — кривая loss на одних весах
y_signed = 2 * y_mix - 1
svm = LinearSVC(C=1.0, loss="hinge", max_iter=10_000).fit(X_mix, y_mix)
margins = y_signed * (X_mix @ svm.coef_.ravel() + svm.intercept_[0])
hinge_loss = np.maximum(0, 1 - margins).mean()
sq_hinge = (np.maximum(0, 1 - margins) ** 2).mean()
print(f"266 hinge loss={hinge_loss:.3f}, squared hinge={sq_hinge:.3f} (sq сильнее штрафует крупные нарушения)")
