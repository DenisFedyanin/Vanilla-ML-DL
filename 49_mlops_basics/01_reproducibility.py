"""Раздел 49 — MLOps basics. Файл 1: Воспроизводимость экспериментов.

Концепция: Случайность в ML.
Большинство алгоритмов (init весов, shuffle, dropout, bootstrap) использует ГПСЧ.
Без фиксации seed одинаковый код даёт разные результаты — невозможно сравнивать
модели и баги воспроизводить.

Концепция: Seed для numpy.
np.random.default_rng(0) создаёт изолированный генератор; глобальный np.random.seed
тоже работает, но менее безопасен (другие модули могут перезаписать).

Концепция: Seed для python random.
Стандартный random.seed(0) фиксирует встроенный модуль random.
Используется sklearn в shuffle, train_test_split (через random_state).

Концепция: random_state в sklearn.
Каждый эстиматор/функция принимает random_state. Передавая один и тот же,
получаешь идентичный результат: split, KMeans, RandomForest и т.д.

Концепция: Версионирование окружения.
Чтобы воспроизвести результат через год — фиксируй версии (requirements.txt,
pip freeze, conda env export, Docker image, lock-файлы). Иначе апдейт sklearn
поменяет floating-point поведение, и метрики поплывут.

Концепция: Источники невоспроизводимости.
GPU-недетерминизм (cuDNN), параллелизм (порядок суммирования float),
hash randomization (PYTHONHASHSEED), внешние данные — всё должно быть зафиксировано.
"""
import random

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def run(seed):
    rng = np.random.default_rng(seed)
    random.seed(seed)
    X, y = make_classification(n_samples=300, n_features=10, random_state=seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed)
    clf = RandomForestClassifier(n_estimators=20, random_state=seed).fit(Xtr, ytr)
    noise = rng.normal(size=5).round(4)
    return clf.score(Xte, yte), noise


a_score, a_noise = run(0)
b_score, b_noise = run(0)
c_score, c_noise = run(1)

print(f"run(seed=0) score={a_score:.4f} noise={a_noise}")
print(f"run(seed=0) score={b_score:.4f} noise={b_noise}  (повтор — идентично)")
print(f"run(seed=1) score={c_score:.4f} noise={c_noise}  (другой seed — другое)")

# Версии окружения — фиксируем для воспроизводимости.
print(f"numpy={np.__version__}")
print("Совет: pip freeze > requirements.txt и зафиксировать PYTHONHASHSEED")
