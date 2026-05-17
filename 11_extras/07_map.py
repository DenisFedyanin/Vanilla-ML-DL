"""Концепция 114: MAP - Maximum A Posteriori.

MAP = MLE + априорное распределение на параметры.
posterior ~ likelihood * prior. Максимизируем posterior.
"""
import numpy as np

# Подбрасываем монетку 10 раз, выпало 7 орлов. Какова p?
heads, n = 7, 10

# MLE: p_hat = 7/10
p_mle = heads / n

# MAP с Beta(2,2) prior (мягкое 'p около 0.5')
# Beta(a,b) prior + biased binomial likelihood -> Beta(a+heads, b+(n-heads))
a, b = 2, 2
p_map = (heads + a - 1) / (n + a + b - 2)

print(f"MLE p = {p_mle:.3f}")
print(f"MAP p (prior Beta(2,2)) = {p_map:.3f}  - 'тянется' к 0.5")
