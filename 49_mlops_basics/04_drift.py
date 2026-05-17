"""Раздел 49 — MLOps. Файл 4: Drift — изменение распределений во времени.

Концепция: Concept drift vs Data drift.
Data (covariate) drift — P(X) меняется (новая аудитория). Concept drift —
P(y|X) меняется (старые признаки уже не объясняют поведение).

Концепция: Label drift (prior shift).
P(y) сместилось — доля классов в проде иная, чем в train. Модель калибрована
под старое распределение и даёт смещённые вероятности.

Концепция: Population Stability Index (PSI).
Разбиваем фичу на бины по train. Считаем долю в train и в новых данных.
PSI = sum( (p_new - p_train) * log(p_new / p_train) ). <0.1 ок, 0.1-0.25 watch, >0.25 alert.

Концепция: Kolmogorov-Smirnov для дрейфа.
KS-статистика = max|F_train(x) - F_new(x)|. Чувствительна к сдвигам распределений.
Альтернатива PSI для непрерывных фич.

Концепция: Detect & React.
Регулярно перепроверяем PSI/KS по каждой фиче и распределению предсказаний.
При тревоге — переобучить, расширить train, добавить новые фичи или включить
adaptive learning.
"""
import numpy as np

rng = np.random.default_rng(0)

# Тренировочное и продовое распределения
train = rng.normal(0, 1, size=5000)
prod_drifted = rng.normal(0.5, 1.2, size=5000)
prod_same = rng.normal(0, 1, size=5000)


def psi(a, b, bins=10):
    edges = np.quantile(a, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    pa, _ = np.histogram(a, edges)
    pb, _ = np.histogram(b, edges)
    pa = pa / pa.sum() + 1e-6
    pb = pb / pb.sum() + 1e-6
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def ks(a, b):
    grid = np.sort(np.concatenate([a, b]))
    fa = np.searchsorted(np.sort(a), grid, side="right") / len(a)
    fb = np.searchsorted(np.sort(b), grid, side="right") / len(b)
    return float(np.max(np.abs(fa - fb)))


print(f"PSI(train, prod_same)    = {psi(train, prod_same):.4f}  (ok, <0.1)")
print(f"PSI(train, prod_drifted) = {psi(train, prod_drifted):.4f}  (alert >0.25)")
print(f"KS (train, prod_same)    = {ks(train, prod_same):.4f}")
print(f"KS (train, prod_drifted) = {ks(train, prod_drifted):.4f}  (значительный сдвиг)")

# Label drift: было 20% положительных, стало 50%
y_train = (rng.uniform(size=2000) < 0.2).astype(int)
y_prod = (rng.uniform(size=2000) < 0.5).astype(int)
print(f"Label drift: P(y=1)_train={y_train.mean():.2f}  P(y=1)_prod={y_prod.mean():.2f}")

# Concept drift (P(y|X) поменялся): тот же X, новая зависимость
X = rng.normal(size=(2000, 2))
y_old = (X[:, 0] > 0).astype(int)
y_new = (X[:, 1] > 0).astype(int)
print(f"Concept drift: corr(y_old,y_new)={np.corrcoef(y_old, y_new)[0,1]:.3f}  (близко к 0 — связь поменялась)")
