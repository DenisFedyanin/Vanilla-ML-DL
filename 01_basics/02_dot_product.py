"""Концепция 2: Скалярное произведение (dot product).

Лежит в основе всех нейронок: y = w·x + b.
"""
import numpy as np

a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, -5.0, 6.0])

dot_manual = sum(x * y for x, y in zip(a, b))
dot_numpy = np.dot(a, b)
dot_op = a @ b

print("Вручную:", dot_manual)
print("np.dot :", dot_numpy)
print("a @ b  :", dot_op)
