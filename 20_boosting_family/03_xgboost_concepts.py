"""Раздел 20 — Семейство бустингов. Файл 3: XGBoost — концепты.

Концепция 610: Регуляризованная функция потерь XGBoost.
Obj = sum_i L(y_i, F(x_i)) + sum_t Omega(h_t),
Omega(h) = gamma * T + 0.5 * lambda * sum_j w_j^2,
где T — число листьев, w_j — значение в листе. gamma штрафует за лишние листья,
lambda — L2 на значения листьев.

Концепция 611: Второй порядок (Newton step).
XGBoost разлагает L по Тейлору 2-го порядка вокруг F_{m-1}:
L(y, F+h) ≈ L(y,F) + g*h + 0.5*H*h^2.
g — градиент, H — гессиан. Оптимум по листу: w*_j = -G_j / (H_j + lambda).

Концепция 612: Выигрыш сплита (gain).
Gain = 0.5 * (G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - G^2/(H+lambda)) - gamma.
Если gain < 0 — сплит не делается (pre-pruning через gamma).

Концепция 613: Псевдокод одного дерева (упрощённо).
  для каждого узла:
    выбрать признак и порог, максимизирующий Gain
    если max_gain <= 0: лист, w = -G/(H+lambda)
    иначе разделить и рекурсивно

Концепция 614: Демонстрация — sklearn GradientBoostingClassifier как прокси.
XGBoost не входит в стандартный sklearn, но GradientBoostingClassifier дает
схожее поведение (первого порядка). Сравним при разных lambda-аналогах
(min_samples_leaf, max_depth).
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier

rng = np.random.default_rng(0)

# === 610-612: численный пример Newton-шага и gain ===
# Пусть в листе 4 объекта; берем g, h для log-loss.
y_true = np.array([1, 0, 1, 1])
F_prev = np.array([0.2, -0.1, 0.0, 0.5])
p = 1 / (1 + np.exp(-F_prev))
g = p - y_true            # градиент log-loss по logit
h = p * (1 - p)           # гессиан
lam = 1.0
gamma = 0.0
G, H = g.sum(), h.sum()
w_star = -G / (H + lam)
print(f"610-611 g={g.round(3)}, h={h.round(3)}, optimal leaf w*={w_star:.3f}")

# Сплит на 2 подмножества: [0,1] vs [2,3]
GL, HL = g[:2].sum(), h[:2].sum()
GR, HR = g[2:].sum(), h[2:].sum()
gain = 0.5 * (GL**2 / (HL + lam) + GR**2 / (HR + lam)
              - G**2 / (H + lam)) - gamma
print(f"612 Gain сплита [0,1]|[2,3]: {gain:.4f} "
      f"(>0 — делать сплит, <0 — нет)")

# === 613: краткий псевдокод печатаем ===
print("613 Псевдокод:")
print("    pred = F0; for m in 1..M: g,h = grad,hess(L,y,pred);")
print("    tree = build_tree(X,g,h,lambda,gamma); pred += nu * tree(X)")

# === 614: демо на синтетике ===
X, y = make_classification(n_samples=300, n_features=8, random_state=0)
for md in (2, 4, 6):
    m = GradientBoostingClassifier(n_estimators=50, max_depth=md,
                                   learning_rate=0.1, random_state=0).fit(X, y)
    print(f"614 GB (proxy XGBoost) max_depth={md}: acc={m.score(X, y):.3f}")
