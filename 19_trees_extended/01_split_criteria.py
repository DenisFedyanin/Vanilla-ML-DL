"""Раздел 19 — Деревья (расширенно).

Файл 1: критерии выбора расщепления.

Концепция 274: Gini impurity.
Gini = 1 - sum p_k^2. Вероятность того, что случайный объект будет
неправильно классифицирован, если предсказывать по распределению классов.
Используется в CART (по умолчанию в sklearn).

Концепция 275: Entropy.
H = -sum p_k log2 p_k. Информационная мера разнообразия классов. Прирост
информации (IG) = H(parent) - sum (n_child/n) H(child). Использует ID3/C4.5.

Концепция 276: MSE (variance) для регрессии.
В листе — среднее y. Импьюрити = дисперсия в узле. Расщепление минимизирует
средневзвешенную дисперсию в дочерних узлах.

Концепция 277: MAE для регрессии.
В листе — медиана y. Импьюрити = средняя |y - median|. Робастнее MSE
к выбросам, но дороже.

Концепция 278: Variance reduction.
Прирост = Var(parent) - sum (n_child/n) Var(child). Та же самая идея, что
и MSE-split — здесь явно как "сколько дисперсии срезали".
"""
import numpy as np

y_cls = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 1])  # 10 объектов, 2 класса
y_reg = np.array([1.0, 1.2, 1.5, 5.0, 5.2, 4.9, 5.1, 5.3, 1.1, 4.8])
feature = np.array([1.0, 1.1, 1.2, 5.0, 5.1, 4.9, 5.0, 5.2, 1.0, 4.8])
threshold = 3.0
left = feature < threshold
right = ~left


def gini(y):
    _, c = np.unique(y, return_counts=True)
    p = c / c.sum()
    return 1 - (p ** 2).sum()


def entropy(y):
    _, c = np.unique(y, return_counts=True)
    p = c / c.sum()
    return -(p * np.log2(p + 1e-12)).sum()


# 274: Gini
g_p, g_l, g_r = gini(y_cls), gini(y_cls[left]), gini(y_cls[right])
nL, nR, N = left.sum(), right.sum(), len(y_cls)
g_split = nL / N * g_l + nR / N * g_r
print(f"274 Gini: parent={g_p:.3f}, after split={g_split:.3f}, gain={g_p-g_split:.3f}")

# 275: Entropy
h_p = entropy(y_cls)
h_split = nL / N * entropy(y_cls[left]) + nR / N * entropy(y_cls[right])
print(f"275 Entropy: parent={h_p:.3f}, after={h_split:.3f}, IG={h_p-h_split:.3f}")

# 276: MSE
v_p = y_reg.var()
v_split = nL / N * y_reg[left].var() + nR / N * y_reg[right].var()
print(f"276 MSE: parent={v_p:.3f}, after={v_split:.3f}, gain={v_p-v_split:.3f}")

# 277: MAE
def mae_impurity(y):
    med = np.median(y)
    return np.abs(y - med).mean()


mae_p = mae_impurity(y_reg)
mae_split = nL / N * mae_impurity(y_reg[left]) + nR / N * mae_impurity(y_reg[right])
print(f"277 MAE: parent={mae_p:.3f}, after={mae_split:.3f}, gain={mae_p-mae_split:.3f}")

# 278: variance reduction = то же, что MSE-gain
print(f"278 Variance reduction = {v_p - v_split:.3f} (та же самая величина)")
