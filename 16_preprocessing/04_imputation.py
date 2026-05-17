"""Раздел 16 — Препроцессинг.

Файл 4: заполнение пропусков (imputation).

Концепция 244: SimpleImputer.
Заполняем NaN константой/средним/медианой/модой по столбцу. Самый простой
и универсальный способ. Не учитывает взаимосвязи признаков.

Концепция 245: KNNImputer.
Для каждой строки с пропуском ищем k ближайших соседей (по другим столбцам)
и берём среднее их значений в пропущенной колонке. Учитывает контекст.

Концепция 246: IterativeImputer (концепт).
Цикл: каждый признак с пропусками предсказываем через регрессию по
остальным, обновляем, повторяем до сходимости. По духу — MICE.
"""
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

rng = np.random.default_rng(0)
X = rng.normal(size=(20, 3))
# вставляем NaN
mask = rng.uniform(size=X.shape) < 0.15
X_nan = X.copy()
X_nan[mask] = np.nan
print(f"Всего пропусков: {np.isnan(X_nan).sum()} из {X_nan.size}")

# 244: SimpleImputer
for strat in ("mean", "median", "most_frequent", "constant"):
    si = SimpleImputer(strategy=strat, fill_value=0.0)
    X_filled = si.fit_transform(X_nan)
    print(f"244 SimpleImputer({strat}): col-means after = {X_filled.mean(axis=0).round(3)}")

# 245: KNNImputer
ki = KNNImputer(n_neighbors=3)
X_knn = ki.fit_transform(X_nan)
# восстановили ли? сравним MSE по позициям пропусков
mse_knn = ((X_knn[mask] - X[mask]) ** 2).mean()
print(f"245 KNNImputer(k=3): MSE на пропусках={mse_knn:.4f}")

# 246: IterativeImputer (MICE-style)
ii = IterativeImputer(random_state=0, max_iter=10)
X_it = ii.fit_transform(X_nan)
mse_it = ((X_it[mask] - X[mask]) ** 2).mean()
print(f"246 IterativeImputer: MSE на пропусках={mse_it:.4f}")
