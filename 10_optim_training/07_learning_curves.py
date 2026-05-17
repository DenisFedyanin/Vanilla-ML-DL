"""Концепция 104: Learning curves.

График: ошибка на train и val в зависимости от РАЗМЕРА обучающей выборки.
Помогает понять - нужны больше данных или нужна другая модель.
"""
import numpy as np
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(0)
X = rng.uniform(-3, 3, size=(500, 1))
y = X.ravel() ** 2 + rng.normal(scale=0.3, size=500)
Xtr, ytr = X[:400], y[:400]
Xva, yva = X[400:], y[400:]

for n in [10, 50, 100, 200, 400]:
    m = LinearRegression().fit(Xtr[:n], ytr[:n])
    tr = ((m.predict(Xtr[:n]) - ytr[:n]) ** 2).mean()
    va = ((m.predict(Xva) - yva) ** 2).mean()
    print(f"n={n:>3}: train MSE={tr:.2f}  val MSE={va:.2f}")
print("Линейная модель не может выучить x^2 - кривые сходятся к высокой ошибке.")
