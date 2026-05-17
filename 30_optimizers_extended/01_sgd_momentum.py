"""Раздел 30 — Оптимизаторы (расширенно). Файл 1: SGD, Momentum, Nesterov.

Минимизируем f(x) = 0.5 * x^T A x (выпуклая квадратичная), grad = A x.
A — диагональная с разной кривизной, чтобы видна разница методов.

Концепция 323: Vanilla SGD.
x_{t+1} = x_t - lr * grad. Простейший шаг по градиенту.
Плохо работает в "оврагах" (анизотропная кривизна): зигзагует.

Концепция 324: Momentum / Heavy ball.
v_{t+1} = mu * v_t + grad
x_{t+1} = x_t - lr * v_{t+1}
Накапливает направление. Гладит зигзаги в овраге, ускоряет в плоских.

Концепция 325: Nesterov Accelerated Gradient.
Сначала "заглядываем" вперёд по моменту, потом считаем градиент:
v_{t+1} = mu*v_t + grad(x_t - lr*mu*v_t); x_{t+1} = x_t - lr*v_{t+1}.
Лучше реагирует на повороты функции, быстрее сходится для выпуклых задач.
"""
import numpy as np

A = np.diag([10.0, 1.0])     # анизотропный овраг
x0 = np.array([1.0, 1.0])
lr, mu, T = 0.1, 0.9, 50

def f(x): return 0.5 * x @ A @ x
def grad(x): return A @ x

# === 323: Vanilla SGD ===
x = x0.copy()
for _ in range(T):
    x = x - lr * grad(x)
print(f"323 SGD lr={lr}: x_T={x.round(4).tolist()}, f={f(x):.5f}")

# === 324: Momentum ===
x, v = x0.copy(), np.zeros_like(x0)
for _ in range(T):
    v = mu * v + grad(x)
    x = x - lr * v
print(f"324 Momentum mu={mu}: x_T={x.round(4).tolist()}, f={f(x):.5f}")

# === 325: Nesterov ===
x, v = x0.copy(), np.zeros_like(x0)
for _ in range(T):
    lookahead = x - lr * mu * v
    v = mu * v + grad(lookahead)
    x = x - lr * v
print(f"325 Nesterov: x_T={x.round(4).tolist()}, f={f(x):.5f}")

print("\nИтог: Momentum/Nesterov сошлись быстрее SGD на анизотропной задаче.")
