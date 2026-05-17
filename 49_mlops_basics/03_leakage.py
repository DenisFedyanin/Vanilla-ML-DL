"""Раздел 49 — MLOps. Файл 3: Утечки данных (data leakage).

Концепция: Что такое leakage.
Информация, которой не будет в проде, попадает в train. Метрики на test
завышены — в реальности модель работает хуже. Самая дорогая ошибка ML.

Концепция: Target leakage.
Фича вычислена с использованием целевой переменной (или прокси для неё).
Пример: 'mean target by category' посчитан по всем данным, включая текущую
строку. Модель просто 'считывает' y из фичи.

Концепция: Train-test contamination.
Препроцессинг (scaler, PCA, target encoder) обучен на всём датасете,
включая test. Test 'просочился' в параметры препроцессинга.

Концепция: Future leakage.
В временных рядах: фича использует значения 'из будущего' (rolling mean,
forward fill). На train метрики прекрасные, в проде — крах.

Концепция: Как защищаться.
Pipeline в sklearn — fit на train, transform на test. CV-aware encoders.
Чёткое разделение во времени для TS. Аудит: 'когда эта фича доступна?'
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)
n = 400
X = rng.normal(size=(n, 5))
y = (X[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(int)

# === Target leakage: фича = mean(y по группам), посчитанная на всех данных ===
groups = rng.integers(0, 10, size=n)
mean_y = np.array([y[groups == g].mean() for g in groups])  # утечка!
X_leak = np.column_stack([X, mean_y])

Xtr, Xte, ytr, yte = train_test_split(X_leak, y, test_size=0.3, random_state=0)
clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
print(f"Target leakage: test acc={clf.score(Xte, yte):.3f}  (нереально высокое)")

# Корректно: считать mean_y только по train, мапить на test
Xtr_raw, Xte_raw, ytr_raw, yte_raw, gtr, gte = train_test_split(
    X, y, groups, test_size=0.3, random_state=0
)
g2mean = {g: ytr_raw[gtr == g].mean() for g in np.unique(gtr)}
mtr = np.array([g2mean[g] for g in gtr])
mte = np.array([g2mean.get(g, ytr_raw.mean()) for g in gte])
clf = LogisticRegression(max_iter=1000).fit(np.column_stack([Xtr_raw, mtr]), ytr_raw)
print(f"Honest encoding: test acc={clf.score(np.column_stack([Xte_raw, mte]), yte_raw):.3f}")

# === Train-test contamination: scaler fit на всех данных ===
sc = StandardScaler().fit(X)            # утечка статистик test
Xs = sc.transform(X)
Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.3, random_state=0)
acc_bad = LogisticRegression(max_iter=1000).fit(Xtr, ytr).score(Xte, yte)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
sc = StandardScaler().fit(Xtr)          # правильно: только train
acc_ok = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), ytr).score(sc.transform(Xte), yte)
print(f"Scaler fit на всём датасете: acc={acc_bad:.3f}")
print(f"Scaler fit только на train  : acc={acc_ok:.3f}  (разница невелика, но в проде это путь к беде)")
