"""Раздел 24 — Ассоциативные правила. Файл 2: метрики правил.

Концепция 673: Support.
sup(X) = P(X встречается в транзакции) = count(X)/N.

Концепция 674: Confidence.
conf(X=>Y) = sup(X∪Y)/sup(X) = P(Y|X). Не учитывает базовую частоту Y.

Концепция 675: Lift.
lift = conf/sup(Y) = P(X∩Y)/(P(X)*P(Y)). =1 — независимы, >1 — положительная,
<1 — отрицательная связь.

Концепция 676: Leverage.
lev = sup(X∪Y) - sup(X)*sup(Y). Аддитивный аналог lift; lev=0 при независимости.

Концепция 677: Conviction.
conv = (1 - sup(Y)) / (1 - conf(X=>Y)). =1 при независимости; чем больше,
тем сильнее «уверенность» в правиле. Не определена, если conf=1.

Концепция 678: All-confidence.
all_conf(X∪Y) = sup(X∪Y) / max(sup(X), sup(Y)). Симметричная мера;
учитывает обе стороны, штрафует за «массовый» предмет.
"""
import numpy as np

# === 673-678: маленький набор транзакций и метрики правил ===
T = [
    {"a", "b", "c"},
    {"a", "b"},
    {"a", "c"},
    {"a"},
    {"b", "c"},
    {"a", "b", "c"},
    {"c"},
    {"a", "b"},
]
N = len(T)

def sup(X):
    return sum(1 for t in T if X.issubset(t)) / N

rules = [
    ({"a"}, {"b"}),
    ({"a"}, {"c"}),
    ({"b"}, {"c"}),
    ({"a", "b"}, {"c"}),
]
for ant, cons in rules:
    A, C = set(ant), set(cons)
    sX = sup(A); sY = sup(C); sXY = sup(A | C)
    confv = sXY / sX
    liftv = confv / sY
    lev = sXY - sX * sY
    conv = (1 - sY) / (1 - confv) if confv < 1 else float("inf")
    all_conf = sXY / max(sX, sY)
    print(f"673-678 {A} => {C}:")
    print(f"     sup={sXY:.2f}, conf={confv:.2f}, lift={liftv:.2f}, "
          f"lev={lev:.2f}, conv={conv:.2f}, all_conf={all_conf:.2f}")
