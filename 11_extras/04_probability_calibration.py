"""Концепция 111: Калибровка вероятностей.

Если модель говорит '80% уверенности', то из таких случаев действительно
80% должны быть положительными. Не у всех моделей это так из коробки.
"""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0)
clf = CalibratedClassifierCV(LinearSVC(), method="sigmoid").fit(Xtr, ytr)
probs = clf.predict_proba(Xte)[:, 1]

# Проверка калибровки: бины по уверенности
for lo, hi in [(0, 0.3), (0.3, 0.7), (0.7, 1.0)]:
    mask = (probs >= lo) & (probs < hi)
    if mask.any():
        print(f"prob in [{lo},{hi}): {mask.sum()} примеров, реальная доля 1 = {yte[mask].mean():.2f}")
