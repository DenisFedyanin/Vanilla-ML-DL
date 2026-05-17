"""Раздел 26 — Semi-supervised. Файл 2: Label Propagation и Label Spreading.

Концепция 695: Граф сходства.
Строим веса w_ij = exp(-||x_i - x_j||^2 / sigma) (или по kNN). Идея: близкие
точки должны иметь близкие метки. Это transduction-постановка.

Концепция 696: Label Propagation.
Распространяет метки по графу: F = D^{-1} W F (D — степени). На каждой итерации
размеченные «клампятся», неразмеченные усредняются по соседям. Сходится к
гармоническому решению.

Концепция 697: Label Spreading.
Похож, но использует нормализованный лапласиан и допускает «забывать»
исходные метки (параметр alpha). Более устойчив к шуму в разметке.
"""
import numpy as np
from sklearn.datasets import make_moons
from sklearn.semi_supervised import LabelPropagation, LabelSpreading

rng = np.random.default_rng(0)
X, y = make_moons(n_samples=200, noise=0.1, random_state=0)

# === 695: оставим 10 меток, остальные — -1 (unlabeled) ===
labels = np.full(len(y), -1)
labeled_idx = rng.choice(len(y), 10, replace=False)
labels[labeled_idx] = y[labeled_idx]
print(f"695 размечено {len(labeled_idx)}/{len(y)} точек "
      f"(остальное = -1 = unlabeled)")

# === 696: Label Propagation ===
lp = LabelPropagation(kernel="knn", n_neighbors=10).fit(X, labels)
acc_lp = (lp.transduction_ == y).mean()
print(f"696 LabelPropagation: transduction accuracy={acc_lp:.3f}")

# === 697: Label Spreading ===
ls = LabelSpreading(kernel="knn", n_neighbors=10, alpha=0.2).fit(X, labels)
acc_ls = (ls.transduction_ == y).mean()
print(f"697 LabelSpreading: transduction accuracy={acc_ls:.3f}")

# Для сравнения — KNN только на размеченных
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3).fit(X[labeled_idx], y[labeled_idx])
acc_knn = knn.score(X, y)
print(f"     baseline KNN на {len(labeled_idx)} метках: accuracy={acc_knn:.3f}")
