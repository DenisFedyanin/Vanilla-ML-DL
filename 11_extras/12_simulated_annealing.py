"""Концепция 119: Simulated Annealing - метод отжига.

Случайно меняем решение. Лучшее принимаем всегда, худшее - с вероятностью
exp(-dE/T). Температура T постепенно падает - алгоритм 'застывает' в оптимуме.
"""
import math
import random

random.seed(0)


def f(x):  # сложная функция с локальным минимумом около -3 и глобальным около 5
    return (x - 5) ** 2 + 8 * math.sin(2 * x)


x = -5.0
T = 5.0
best, best_v = x, f(x)
for it in range(5000):
    x_new = x + random.gauss(0, 0.5)
    df = f(x_new) - f(x)
    if df < 0 or random.random() < math.exp(-df / T):
        x = x_new
        if f(x) < best_v:
            best, best_v = x, f(x)
    T *= 0.9995

print(f"Лучший найденный x = {best:.3f}, f(x) = {best_v:.3f}")
