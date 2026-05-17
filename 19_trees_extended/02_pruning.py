"""Раздел 19 — Деревья (расширенно).

Файл 2: обрезка деревьев.

Концепция 279: Pre-pruning (предобрезка).
Останавливаем рост заранее: max_depth, min_samples_leaf, min_impurity_decrease.
Простой и эффективный способ против переобучения.

Концепция 280: Cost-complexity pruning (post-pruning, alpha).
Растим полное дерево, затем "обрезаем" его, минимизируя R(T) + alpha*|T|, где
R(T) — ошибка, |T| — число листьев. ccp_path вычисляет точки эффективных alpha.

Концепция 281: REP (Reduced Error Pruning, концепт).
Используем валидационную выборку. Для каждой внутренней вершины: если
схлопнуть в лист, валидационная ошибка не растёт — схлопываем. Простой и
быстрый бэкенд для post-pruning.
"""
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = make_moons(n_samples=400, noise=0.3, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

# 279: pre-pruning
deep = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)
prep = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=0).fit(X_tr, y_tr)
print(f"279 unlimited: train={deep.score(X_tr, y_tr):.3f}, test={deep.score(X_te, y_te):.3f}, n_leaves={deep.get_n_leaves()}")
print(f"     pre-prune (depth=4, leaf=10): train={prep.score(X_tr, y_tr):.3f}, test={prep.score(X_te, y_te):.3f}, n_leaves={prep.get_n_leaves()}")

# 280: cost-complexity pruning
full = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)
path = full.cost_complexity_pruning_path(X_tr, y_tr)
alphas = path.ccp_alphas
best_alpha = 0.0
best_score = 0.0
for a in alphas[::max(1, len(alphas) // 8)]:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=a).fit(X_tr, y_tr)
    s = clf.score(X_te, y_te)
    if s > best_score:
        best_score, best_alpha = s, a
clf_best = DecisionTreeClassifier(random_state=0, ccp_alpha=best_alpha).fit(X_tr, y_tr)
print(f"280 ccp_alpha best={best_alpha:.5f}: test={best_score:.3f}, n_leaves={clf_best.get_n_leaves()}")

# 281: REP (упрощённая идея) — ручной обход по post-order и схлопывание,
# если валидационная ошибка не растёт. Покажем концепт.
tree_ = full.tree_
val_pred_full = full.predict(X_te)
val_err_full = (val_pred_full != y_te).mean()
print(f"281 REP concept: исходн. валидац. ошибка полного дерева={val_err_full:.3f}, листьев={full.get_n_leaves()}")
print(f"     идея: пройти post-order, заменять поддерево на лист с мажоритарным классом,")
print(f"     если ошибка на валидации не возрастает — оставлять схлопнутое")
