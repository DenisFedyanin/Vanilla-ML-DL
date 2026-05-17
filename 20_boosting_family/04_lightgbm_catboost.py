"""Раздел 20 — Семейство бустингов. Файл 4: LightGBM и CatBoost (концепты).

Концепция 615: Histogram-based splits.
Вместо перебора всех уникальных значений признака — делим в гистограмму
(255 бинов). Ищем сплит между бинами. На порядок быстрее и почти не теряет
качество. Используется в LightGBM, HistGradientBoostingClassifier.

Концепция 616: GOSS — Gradient-based One-Side Sampling.
Объекты с большим |g| (хуже подогнаны) сохраняем все; из остальных берём
случайную долю. Сохраняет точное распределение градиентов в важной части,
ускоряет на 2-3 раза.

Концепция 617: EFB — Exclusive Feature Bundling.
Если два разреженных признака почти не пересекаются по ненулям, склеиваем
их в один (с разными «слотами»). Снижает размер признакового пространства
для очень разреженных табличек (one-hot и т.п.).

Концепция 618: Leaf-wise (best-first) рост дерева.
В обычном GB дерево растёт level-wise (по слоям). LightGBM выбирает лист с
максимальным gain и растит его — глубже там, где данные сложнее. Часто даёт
лучшее качество, но риск переобучения при маленьких выборках.

Концепция 619: CatBoost — Ordered Target Statistics.
Категориальные признаки кодируются «таргет-средним», но усреднение считается
только по объектам, идущим в случайной перестановке РАНЬШЕ текущего.
Это убирает target leakage обычного mean-encoding.

Концепция 620: CatBoost — Ordered Boosting.
Аналогично, на каждом шаге для предсказания i-го объекта используются деревья,
обученные без него. Дороже, но избавляет от «prediction shift».
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier

rng = np.random.default_rng(0)

# === 615: HistGB vs обычный GB ===
X, y = make_classification(n_samples=800, n_features=10, random_state=0)
hgb = HistGradientBoostingClassifier(max_iter=30, random_state=0).fit(X, y)
gb = GradientBoostingClassifier(n_estimators=30, random_state=0).fit(X, y)
print(f"615 HistGB acc={hgb.score(X, y):.3f}, обычный GB acc={gb.score(X, y):.3f}")

# === 616: имитация GOSS ===
# Допустим, у нас 1000 объектов с градиентами; берём топ-20% по |g|, остальные семплим 30%.
g = rng.normal(size=1000)
top_idx = np.argsort(-np.abs(g))[: int(0.2 * len(g))]
rest_idx = np.setdiff1d(np.arange(len(g)), top_idx)
sample_rest = rng.choice(rest_idx, size=int(0.3 * len(rest_idx)), replace=False)
sel = np.concatenate([top_idx, sample_rest])
print(f"616 GOSS-выборка: всего {len(g)}, осталось {len(sel)} "
      f"({100*len(sel)/len(g):.0f}%)")

# === 617: EFB-иллюстрация ===
# Два разреженных признака почти не пересекаются по ненулям → один новый признак.
f1 = np.array([0, 0, 3, 0, 0, 0, 7, 0])
f2 = np.array([2, 1, 0, 0, 5, 0, 0, 0])
conflict = ((f1 != 0) & (f2 != 0)).sum()
bundle = np.where(f1 != 0, f1, f2)  # склейка
print(f"617 EFB: конфликтов={conflict}, склеенный признак={bundle.tolist()}")

# === 618: leaf-wise — глубже там, где gain больше (имитация) ===
gains = [3.2, 1.1, 2.7, 0.5, 4.0]
order = np.argsort(-np.array(gains))
print(f"618 Порядок расширения листьев (best-first): {order.tolist()}")

# === 619-620: CatBoost-концепты печатаем + ordered TS demo ===
# Случайная перестановка → таргет-среднее только по «прошлым» объектам.
y_cat = np.array([1, 0, 1, 1, 0, 1, 0, 0])
cat = np.array(["a", "b", "a", "a", "b", "a", "b", "a"])
perm = rng.permutation(len(y_cat))
enc = np.zeros(len(y_cat))
for pos, idx in enumerate(perm):
    past = perm[:pos]
    same = past[cat[past] == cat[idx]]
    enc[idx] = y_cat[same].mean() if len(same) else 0.5
print(f"619 Ordered Target Statistics для cat={cat.tolist()}:")
print(f"     enc={enc.round(2).tolist()}")
print("620 Ordered Boosting: для предсказания i-го объекта используем деревья, "
      "обученные без него (по случайной перестановке)")
