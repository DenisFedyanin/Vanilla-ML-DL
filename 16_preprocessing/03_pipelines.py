"""Раздел 16 — Препроцессинг.

Файл 3: пайплайны и предотвращение data leakage.

Концепция 240: Pipeline.
Цепочка (трансформер, ..., трансформер, оценщик). fit/predict проходят
сквозь все шаги. Гарантия, что препроцессинг применяется одинаково к
train и test.

Концепция 241: ColumnTransformer.
Применяет разные трансформации к разным столбцам (числовые ↔ категориальные).
Соединяется с финальным эстиматором через Pipeline.

Концепция 242: FunctionTransformer.
Оборачивает произвольную функцию в трансформер (например, log1p) для
использования в Pipeline без написания класса.

Концепция 243: Data leakage из-за неправильного порядка.
Если scaler.fit на ВСЕХ данных (включая test), статистика теста "утекает"
в train -> завышенная метрика. Правильно: fit только на train, transform — на оба.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

rng = np.random.default_rng(0)
n = 300
age = rng.uniform(18, 70, n)
income = rng.lognormal(10, 1, n)
city = rng.choice(["msk", "spb", "kzn"], size=n)
y = ((age / 70 + np.log(income) / 12 + (city == "msk") * 0.2 + rng.normal(0, 0.3, n)) > 1.5).astype(int)

X_num = np.column_stack([age, income])
X_cat = city.reshape(-1, 1)
X = np.column_stack([age, income, city]).astype(object)

# 240/241: ColumnTransformer + Pipeline
preproc = ColumnTransformer(
    [
        ("num", StandardScaler(), [0, 1]),
        ("cat", OneHotEncoder(), [2]),
    ]
)
pipe = Pipeline([("prep", preproc), ("lr", LogisticRegression(max_iter=200))])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
pipe.fit(X_tr, y_tr)
print(f"240/241 Pipeline+CT accuracy={pipe.score(X_te, y_te):.3f}")

# 242: FunctionTransformer — log1p числовой колонки
log_pipe = Pipeline(
    [
        ("log", FunctionTransformer(np.log1p, validate=True)),
        ("scaler", StandardScaler()),
    ]
)
X_num_log = log_pipe.fit_transform(X_num)
print(f"242 FunctionTransformer(log1p)+scale: shape={X_num_log.shape}, mean={X_num_log.mean(axis=0).round(3)}")

# 243: data leakage demo
# WRONG: fit на всех данных
scaler_bad = StandardScaler().fit(X_num)
mean_bad = scaler_bad.mean_
# RIGHT: только на train
X_num_tr, X_num_te = train_test_split(X_num, test_size=0.3, random_state=0)
scaler_good = StandardScaler().fit(X_num_tr)
print(f"243 LEAKAGE: scaler fit на all  -> mean_age={mean_bad[0]:.2f}")
print(f"     CORRECT: scaler fit на tr  -> mean_age={scaler_good.mean_[0]:.2f}")
print(f"     test mean при правильном scaler не равен 0: {scaler_good.transform(X_num_te).mean(axis=0).round(3)}")
