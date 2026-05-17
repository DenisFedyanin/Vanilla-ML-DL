"""Раздел 15 — Feature Engineering.

Файл 2: кодирование КАТЕГОРИАЛЬНЫХ признаков.

Концепция 206: One-Hot Encoding (recap).
Каждая категория -> отдельный 0/1-столбец. Базовый способ, не масштабируется
при сотнях тысяч уникальных значений.

Концепция 207: Label Encoding.
Категория -> целое число 0..k-1. Уместно для tree-based моделей и для
ordinal признаков; для линейных создаст ложный порядок.

Концепция 208: Ordinal Encoding.
То же, но порядок задаётся вручную по смыслу (low<med<high).

Концепция 209: Target Encoding.
Категория -> среднее target по этой категории. Очень мощно, но течёт цель.
Делать ТОЛЬКО на train, со сглаживанием и в out-of-fold режиме.

Концепция 210: Frequency Encoding.
Категория -> её частота встречаемости. Полезно когда сама редкость значима.

Концепция 211: Leave-One-Out Encoding.
Среднее target по категории БЕЗ текущей строки. Слабее течёт цель, чем
обычный target encoding.

Концепция 212: Weight of Evidence (WoE).
ln(P(x|y=1) / P(x|y=0)). Стандарт в кредитном скоринге, превращает
категорию в монотонный логит-фактор.

Концепция 213: Hashing trick.
hash(category) mod m -> номер столбца. Нет хранения словаря, есть коллизии.
Идеально для онлайн-обучения и текстовых n-gram признаков.
"""
import numpy as np

cats = np.array(["A", "B", "A", "C", "B", "B", "A", "C", "C", "C"])
y = np.array([1, 0, 1, 1, 0, 0, 1, 1, 0, 1])

uniq = np.unique(cats)

# 206: OHE
ohe = (cats[:, None] == uniq[None, :]).astype(int)
print(f"206 OHE shape={ohe.shape}, head=\n{ohe[:3]}")

# 207: Label
mapping = {c: i for i, c in enumerate(uniq)}
label = np.array([mapping[c] for c in cats])
print(f"207 Label: {label}")

# 208: Ordinal с заданным порядком
order = {"A": 0, "B": 1, "C": 2}
ordinal = np.array([order[c] for c in cats])
print(f"208 Ordinal (custom order): {ordinal}")

# 209: Target encoding с сглаживанием
global_mean = y.mean()
alpha = 5  # сила сглаживания
te = np.zeros_like(y, dtype=float)
for c in uniq:
    mask = cats == c
    n_c = mask.sum()
    mean_c = y[mask].mean()
    smoothed = (n_c * mean_c + alpha * global_mean) / (n_c + alpha)
    te[mask] = smoothed
print(f"209 TargetEnc (alpha={alpha}): {te.round(3)}")

# 210: Frequency
freq_map = {c: (cats == c).mean() for c in uniq}
freq = np.array([freq_map[c] for c in cats])
print(f"210 FreqEnc: {freq.round(2)}")

# 211: Leave-One-Out
loo = np.zeros_like(y, dtype=float)
for i, c in enumerate(cats):
    mask = (cats == c) & (np.arange(len(cats)) != i)
    loo[i] = y[mask].mean() if mask.any() else global_mean
print(f"211 LOO: {loo.round(3)}")

# 212: WoE
woe_map = {}
eps = 1e-3
n_pos = y.sum()
n_neg = len(y) - n_pos
for c in uniq:
    mask = cats == c
    p_pos = max(y[mask].sum() / n_pos, eps)
    p_neg = max((1 - y[mask]).sum() / n_neg, eps)
    woe_map[c] = np.log(p_pos / p_neg)
woe = np.array([woe_map[c] for c in cats])
print(f"212 WoE: {{A: {woe_map['A']:.3f}, B: {woe_map['B']:.3f}, C: {woe_map['C']:.3f}}}")

# 213: Hashing trick
m = 4
hashed = np.array([hash(c) % m for c in cats])
print(f"213 Hashing(m={m}): {hashed}")
