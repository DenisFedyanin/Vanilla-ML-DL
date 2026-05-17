"""Раздел 16 — Препроцессинг.

Файл 1: масштабирование признаков.

Концепция 230: StandardScaler.
(x - mean)/std. Превращает признак в N(0,1)-подобный. База для линейных
моделей, SVM, нейросетей, PCA.

Концепция 231: MinMaxScaler.
(x - min)/(max - min). В [0,1]. Для алгоритмов, чувствительных к границам
(нейронные сети с sigmoid, изображения).

Концепция 232: MaxAbsScaler.
x / max|x|. В [-1, 1]. Сохраняет знак и разреженность (sparse matrices).

Концепция 233: RobustScaler.
(x - median)/IQR. Устойчив к выбросам, т.к. медиана и IQR не "уезжают"
от единичного аномального значения.

Концепция 234: Normalizer L2.
Каждая СТРОКА (объект) нормируется в единичную L2-норму. Стандарт перед
косинусным сходством.

Концепция 235: Normalizer L1.
Каждая строка нормируется в единичную L1-норму (сумма = 1). Удобно
интерпретировать как распределение долей.

Концепция 236: Binarizer.
x > threshold -> 1, иначе 0. Простая дискретизация одной чертой.
"""
import numpy as np
from sklearn.preprocessing import (
    Binarizer,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    RobustScaler,
    StandardScaler,
)

rng = np.random.default_rng(0)
X = rng.normal(loc=5, scale=2, size=(8, 3))
X[0, 0] = 100  # выброс

# 230: StandardScaler (а ниже — ручной вариант)
ss = StandardScaler().fit(X)
X_ss = ss.transform(X)
manual = (X - X.mean(axis=0)) / X.std(axis=0)
print(f"230 StandardScaler: mean per col≈{X_ss.mean(axis=0).round(3)}, std≈{X_ss.std(axis=0).round(3)}")
print(f"     manual vs sklearn close? {np.allclose(manual, X_ss)}")

# 231: MinMax
mm = MinMaxScaler().fit_transform(X)
print(f"231 MinMax: min={mm.min(axis=0).round(3)}, max={mm.max(axis=0).round(3)}")

# 232: MaxAbs
ma = MaxAbsScaler().fit_transform(X)
print(f"232 MaxAbs: max|.| per col={np.abs(ma).max(axis=0).round(3)}")

# 233: Robust (median, IQR) — выброс почти не сдвигает шкалу
rs = RobustScaler().fit_transform(X)
print(f"233 Robust: median={np.median(rs, axis=0).round(3)}, IQR={(np.percentile(rs,75,axis=0)-np.percentile(rs,25,axis=0)).round(3)}")

# 234: Normalizer L2 (по строкам)
nl2 = Normalizer(norm="l2").fit_transform(X)
print(f"234 NormalizerL2: row norms={np.linalg.norm(nl2, axis=1).round(3)}")

# 235: Normalizer L1
nl1 = Normalizer(norm="l1").fit_transform(np.abs(X))
print(f"235 NormalizerL1: row sums={np.abs(nl1).sum(axis=1).round(3)}")

# 236: Binarizer
b = Binarizer(threshold=5.0).fit_transform(X)
print(f"236 Binarizer(thr=5): first row={b[0]} -> {b[0].astype(int)}")
