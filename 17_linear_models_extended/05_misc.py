"""Раздел 17 — Линейные модели (расширенно).

Файл 5: разные специальные регрессии.

Концепция 259: PassiveAggressive.
Онлайн-алгоритм: при малой ошибке не обновляемся (passive), при большой —
обновляем веса агрессивно настолько, чтобы текущий пример стал верным.

Концепция 260: SGDRegressor.
Стохастический градиентный спуск на линейной модели, с любым штрафом
(l1/l2/elasticnet) и любой loss (squared, huber, epsilon_insensitive).

Концепция 261: OrthogonalMatchingPursuit (OMP).
Жадно добавляем по одному "наиболее коррелированному" с остатком признаку,
пока не наберём заданное число ненулевых коэффициентов. Дешёвая sparse-регрессия.

Концепция 262: LARS (Least Angle Regression).
Эффективный путь решения Lasso: на каждом шаге увеличиваем коэффициент
самого "согласованного" с остатком признака, пока другой не сравняется.

Концепция 263: Isotonic regression.
Подгоняем монотонную (неубывающую) ступенчатую функцию. Удобно для
калибровки вероятностей классификатора (Platt vs Isotonic).
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Lars, OrthogonalMatchingPursuit, PassiveAggressiveRegressor, SGDRegressor

rng = np.random.default_rng(0)
n, d = 300, 10
X = rng.normal(size=(n, d))
w_true = np.array([2.0, -1.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
y = X @ w_true + rng.normal(0, 0.3, n)

# 259
pa = PassiveAggressiveRegressor(max_iter=500, random_state=0).fit(X, y)
print(f"259 PA-Regressor coefs[:4]={pa.coef_[:4].round(3)}")

# 260
sgd = SGDRegressor(loss="squared_error", penalty="l2", max_iter=1000, random_state=0).fit(X, y)
print(f"260 SGDRegressor coefs[:4]={sgd.coef_[:4].round(3)}")

# 261: OMP с явно заданным числом ненулевых
omp = OrthogonalMatchingPursuit(n_nonzero_coefs=3).fit(X, y)
nz = np.flatnonzero(omp.coef_)
print(f"261 OMP nonzero idx={nz.tolist()}, coefs={omp.coef_[nz].round(3)}")

# 262: LARS path — финальные коэффициенты
lars = Lars(n_nonzero_coefs=5).fit(X, y)
print(f"262 LARS coefs[:4]={lars.coef_[:4].round(3)}")

# 263: Isotonic regression — калибровка немонотонной зашумлённой функции
x_iso = np.linspace(0, 10, 60)
y_iso = np.log1p(x_iso) + rng.normal(0, 0.3, 60)
iso = IsotonicRegression().fit(x_iso, y_iso)
y_pred = iso.predict(x_iso)
diffs = np.diff(y_pred)
print(f"263 Isotonic: монотонна? {(diffs >= -1e-9).all()}, y_pred[:5]={y_pred[:5].round(3)}")
