"""Концепция 6: Цепное правило.

Если y = f(g(x)), то dy/dx = f'(g(x)) * g'(x).
На этом построен backpropagation.
"""
import math

x = 2.0

g = lambda x: 3 * x + 1          # внутренняя
f = lambda u: u ** 2             # внешняя
y = f(g(x))

dg = 3                            # g'(x) = 3
df = 2 * g(x)                     # f'(u) = 2u
dy_dx_chain = df * dg

dy_dx_num = (f(g(x + 1e-6)) - f(g(x - 1e-6))) / 2e-6

print(f"y = {y}")
print(f"dy/dx (по правилу цепи) = {dy_dx_chain}")
print(f"dy/dx (численно)        = {dy_dx_num:.4f}")
