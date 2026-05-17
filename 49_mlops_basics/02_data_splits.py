"""Раздел 49 — MLOps. Файл 2: Стратегии разбиения данных.

Концепция: Train / Validation / Test.
Train — для обучения, Val — для выбора гиперпараметров, Test — финальная оценка.
Test используется один раз; иначе он перестаёт быть честным.

Концепция: Случайное разбиение.
sklearn.train_test_split. Работает для IID данных, но опасно при наличии групп
или временной структуры.

Концепция: Стратифицированное разбиение.
Сохраняет долю классов в train и test. Критично для несбалансированных задач
(redkij класс может полностью попасть в одну часть).

Концепция: Временное разбиение (time-based).
Если данные имеют дату — нельзя обучаться на 'будущем'. Тренируемся на
[начало, t0], тестируем на (t0, конец]. Иначе утечка из будущего.

Концепция: Group-aware split.
Когда наблюдения группируются (пациент, пользователь, документ), нельзя чтобы
одна группа попала и в train, и в test. GroupKFold / GroupShuffleSplit решают.

Концепция: Holdout vs Cross-validation.
Holdout — одно разбиение (быстро). CV — k-fold (точнее, но дороже).
Для маленьких выборок CV даёт более стабильную оценку.
"""
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

rng = np.random.default_rng(0)
n = 200
X = rng.normal(size=(n, 3))
y = (rng.uniform(size=n) < 0.2).astype(int)  # 20% положительных
groups = rng.integers(0, 40, size=n)         # 40 групп
times = np.arange(n)

# Случайное
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
print(f"Random split: train={len(ytr)} test={len(yte)} pos_test={yte.mean():.2f}")

# Стратифицированное
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
print(f"Stratified : pos_train={ytr.mean():.3f} pos_test={yte.mean():.3f} (близко к 0.2)")

# Временное (никаких будущих наблюдений в train)
cut = int(0.7 * n)
tr_idx = times < cut
te_idx = times >= cut
print(f"Time-based : train_max_t={times[tr_idx].max()} test_min_t={times[te_idx].min()}")

# Group-aware: одна группа целиком в train или в test
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
tr_i, te_i = next(gss.split(X, y, groups=groups))
overlap = set(groups[tr_i]) & set(groups[te_i])
print(f"Group split: |train_groups|={len(set(groups[tr_i]))} overlap={len(overlap)} (должно быть 0)")
