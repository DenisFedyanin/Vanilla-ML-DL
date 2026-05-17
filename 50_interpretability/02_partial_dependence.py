"""Раздел 50 — Interpretability. Файл 2: Partial Dependence / ICE / ALE.

Концепция: Partial Dependence Plot (PDP).
Как меняется среднее предсказание модели при варьировании одной фичи,
если усреднить по всем остальным. Формально: PD_j(v) = E_{X_{-j}}[f(v, X_{-j})].
Реализация: подменяем X[:, j] := v, усредняем f(X).

Концепция: Достоинства и недостатки PDP.
Простой, модель-агностичный. Но при сильной корреляции фич усреднение происходит
по 'нереальным' комбинациям (например, рост=1м50, вес=120кг) — оценка смещена.

Концепция: ICE (Individual Conditional Expectation).
PDP для каждого объекта отдельно (не усредняем). Видны взаимодействия:
если кривые расходятся 'веером', значит эффект фичи зависит от других.

Концепция: ALE (Accumulated Local Effects).
Усредняет не по всему X, а в локальной окрестности — снижает проблему
коррелированных фич. Считает прирост предсказания в узких бинах фичи.

Концепция: Применение.
PDP/ICE/ALE — стандарт для регуляторных отчётов (банкинг, медицина).
Объясняют 'что произойдёт с предсказанием, если поднять доход на 10%'.
"""
import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor

rng = np.random.default_rng(0)
X, y = make_regression(n_samples=400, n_features=4, n_informative=3, noise=4.0, random_state=0)
model = GradientBoostingRegressor(n_estimators=80, random_state=0).fit(X, y)

j = 0  # анализируем фичу 0
grid = np.quantile(X[:, j], np.linspace(0.05, 0.95, 8))

# PDP
pd_vals = []
for v in grid:
    Xc = X.copy()
    Xc[:, j] = v
    pd_vals.append(model.predict(Xc).mean())
print("PDP (фича 0):")
for v, p in zip(grid, pd_vals):
    print(f"  x={v:+.2f}  E[f]={p:+.2f}")

# ICE для 5 случайных объектов
ids = rng.choice(len(X), size=5, replace=False)
print("ICE (5 объектов): значения f(x) при разных v фичи 0")
for i in ids:
    row = []
    for v in grid:
        x = X[i].copy()
        x[j] = v
        row.append(model.predict(x.reshape(1, -1))[0])
    print(f"  obj{i}: {np.round(row, 1)}")

# ALE: локальные приросты в бинах
edges = np.quantile(X[:, j], np.linspace(0, 1, 8))
ale = [0.0]
for k in range(len(edges) - 1):
    mask = (X[:, j] >= edges[k]) & (X[:, j] <= edges[k + 1])
    if mask.sum() == 0:
        ale.append(ale[-1])
        continue
    Xa = X[mask].copy()
    Xb = X[mask].copy()
    Xa[:, j] = edges[k]
    Xb[:, j] = edges[k + 1]
    ale.append(ale[-1] + (model.predict(Xb) - model.predict(Xa)).mean())
ale = np.array(ale) - np.mean(ale)
print(f"ALE кривая   : {ale.round(2)}  (центрирована на 0)")
