"""Раздел 24 — Ассоциативные правила. Файл 1: Apriori.

Концепция 669: Транзакции и itemset.
Транзакция — множество купленных вместе товаров. Itemset размера k — набор
из k товаров. Цель — найти часто встречающиеся itemsets (frequent itemsets).

Концепция 670: Support (поддержка).
support(X) = (число транзакций, содержащих X) / (общее число транзакций).
Itemset «частый», если support >= min_support.

Концепция 671: Apriori-принцип (anti-monotone).
Если X не частый, то любое его надмножество — тоже не частое.
Это позволяет резко сократить перебор: на шаге k+1 рассматриваем только
itemsets, у которых ВСЕ k-подмножества частые.

Концепция 672: Confidence и Lift для правила X => Y.
confidence(X=>Y) = support(X∪Y) / support(X).
lift(X=>Y) = confidence(X=>Y) / support(Y) — во сколько раз условие X
увеличивает вероятность Y по сравнению с базовой частотой Y. lift > 1 — позитивная связь.
"""
import numpy as np
from itertools import combinations

# === 669: 8 транзакций маленького супермаркета ===
T = [
    {"bread", "milk"},
    {"bread", "diaper", "beer", "egg"},
    {"milk", "diaper", "beer", "cola"},
    {"bread", "milk", "diaper", "beer"},
    {"bread", "milk", "diaper", "cola"},
    {"bread", "milk"},
    {"diaper", "beer"},
    {"bread", "milk", "diaper"},
]
items = sorted({i for t in T for i in t})
N = len(T)
print(f"669 транзакций={N}, уникальных товаров={len(items)}: {items}")

def support(itemset, transactions):
    s = sum(1 for t in transactions if itemset.issubset(t))
    return s / len(transactions)

# === 670: support для L1 ===
min_sup = 0.3
L1 = [frozenset([i]) for i in items if support(frozenset([i]), T) >= min_sup]
print(f"670 L1 (single items, sup>={min_sup}): {[set(x) for x in L1]}")

# === 671: генерируем L2 по Apriori-принципу ===
def gen_candidates(L_prev, k):
    cand = set()
    for a in L_prev:
        for b in L_prev:
            u = a | b
            if len(u) == k:
                # проверка: все (k-1)-подмножества должны быть в L_prev
                if all(frozenset(s) in set(L_prev) for s in combinations(u, k - 1)):
                    cand.add(frozenset(u))
    return list(cand)

L2_cand = gen_candidates(L1, 2)
L2 = [c for c in L2_cand if support(c, T) >= min_sup]
print(f"671 L2 (после Apriori-pruning): {[set(x) for x in L2]}")

L3_cand = gen_candidates(L2, 3)
L3 = [c for c in L3_cand if support(c, T) >= min_sup]
print(f"671 L3: {[set(x) for x in L3]}")

# === 672: confidence и lift для нескольких правил ===
def conf(X, Y):
    return support(X | Y, T) / support(X, T)

def lift(X, Y):
    return conf(X, Y) / support(Y, T)

rules = [
    (frozenset(["diaper"]), frozenset(["beer"])),
    (frozenset(["bread"]),  frozenset(["milk"])),
    (frozenset(["beer"]),   frozenset(["diaper"])),
]
for X, Y in rules:
    print(f"672 {set(X)} => {set(Y)}: "
          f"sup={support(X|Y, T):.2f}, conf={conf(X, Y):.2f}, lift={lift(X, Y):.2f}")
