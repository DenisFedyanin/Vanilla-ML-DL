"""Раздел 25 — Поиск аномалий. Файл 2: модельные методы.

Концепция 684: Isolation Forest.
Случайно делает «случайные сплиты»; аномалии изолируются за меньшее число
сплитов (короткая средняя глубина в дереве). Скор = -avg_path. Быстрый,
работает в высоких размерностях.

Концепция 685: LOF — Local Outlier Factor.
Сравнивает локальную плотность точки с плотностью её соседей. Если у точки
плотность сильно ниже, чем у соседей — LOF > 1, аномалия.

Концепция 686: Elliptic Envelope.
Подгоняет надёжный эллипс (mu, Sigma — Minimum Covariance Determinant) к
большинству данных. Аномалии — точки с большой mahalanobis-дистанцией к этой
оценке. Предполагается приблизительная нормальность.

Концепция 687: One-Class SVM.
Учит границу вокруг «нормальных» данных в kernel-пространстве. Аномалии —
точки с decision_function < 0. Хорош для нелинейных границ нормы.
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM

rng = np.random.default_rng(0)
# 200 нормальных + 10 выбросов
X_norm = rng.normal(0, 1, size=(200, 2))
X_out = rng.uniform(-6, 6, size=(10, 2))
X = np.vstack([X_norm, X_out])
y_true = np.concatenate([np.ones(200), -np.ones(10)])

def stats(pred):
    tp = ((pred == -1) & (y_true == -1)).sum()
    fp = ((pred == -1) & (y_true == 1)).sum()
    return tp, fp

# === 684: Isolation Forest ===
iso = IsolationForest(contamination=0.05, random_state=0).fit(X)
tp, fp = stats(iso.predict(X))
print(f"684 IsolationForest: TP={tp}/10, FP={fp}")

# === 685: LOF ===
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
pred = lof.fit_predict(X)
tp, fp = stats(pred)
print(f"685 LOF: TP={tp}/10, FP={fp}")

# === 686: Elliptic Envelope ===
ee = EllipticEnvelope(contamination=0.05, random_state=0, support_fraction=0.9).fit(X)
tp, fp = stats(ee.predict(X))
print(f"686 EllipticEnvelope: TP={tp}/10, FP={fp}")

# === 687: One-Class SVM ===
oc = OneClassSVM(kernel="rbf", gamma="auto", nu=0.05).fit(X_norm)
tp, fp = stats(oc.predict(X))
print(f"687 OneClassSVM (учился только на нормальных): TP={tp}/10, FP={fp}")
