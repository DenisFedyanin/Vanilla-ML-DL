"""Раздел 60 — Анализ.

Файл 3: Автоматическое дифференцирование (AD).

Концепция: Forward-mode AD через dual numbers.
Дуальное число a + b*eps, eps^2=0. Если x = v + eps, то f(x) = f(v) + f'(v)*eps.
Все базовые операции переопределяются: (a+be)*(c+de) = ac + (ad+bc)e.
Эффективно для функций R -> R^m (один входной направлений за проход).

Концепция: Reverse-mode AD.
Строим вычислительный граф. Прямой проход вычисляет значения, обратный — градиенты
по правилу цепочки в обратном порядке. Один обратный проход дает grad f для R^n -> R.
Используется в backprop.

Концепция: VJP (Vector-Jacobian product).
Reverse-mode AD естественно вычисляет v^T J через обратный проход.
JAX/PyTorch предоставляют vjp как примитив для эффективных производных.
"""
import numpy as np

# === Forward-mode: класс Dual ===
class D:
    def __init__(self, v, d=0.0):
        self.v, self.d = float(v), float(d)
    def __add__(self, o):
        o = o if isinstance(o, D) else D(o)
        return D(self.v + o.v, self.d + o.d)
    __radd__ = __add__
    def __sub__(self, o):
        o = o if isinstance(o, D) else D(o)
        return D(self.v - o.v, self.d - o.d)
    def __rsub__(self, o):
        return D(o) - self
    def __mul__(self, o):
        o = o if isinstance(o, D) else D(o)
        return D(self.v * o.v, self.v * o.d + self.d * o.v)
    __rmul__ = __mul__
    def __truediv__(self, o):
        o = o if isinstance(o, D) else D(o)
        return D(self.v / o.v, (self.d * o.v - self.v * o.d) / (o.v ** 2))
    def __pow__(self, k):
        return D(self.v ** k, k * self.v ** (k - 1) * self.d)
    def sin(self): return D(np.sin(self.v), np.cos(self.v) * self.d)
    def cos(self): return D(np.cos(self.v), -np.sin(self.v) * self.d)
    def exp(self): return D(np.exp(self.v), np.exp(self.v) * self.d)

# Производная f(x) = x^3 + sin(x) в x=1.5
x = D(1.5, 1.0)  # seed: dx/dx = 1
y = x ** 3 + x.sin()
print(f"Forward AD: f(1.5)={y.v:.4f}, f'(1.5)={y.d:.4f}  (теор: 3x^2+cos(x) = {3*1.5**2 + np.cos(1.5):.4f})")

# Градиент по двум переменным: seed по очереди
def grad_forward_2d(f, p):
    g = np.zeros(2)
    for i in range(2):
        a = D(p[0], 1.0 if i == 0 else 0.0)
        b = D(p[1], 1.0 if i == 1 else 0.0)
        g[i] = f(a, b).d
    return g

f2 = lambda a, b: a * a + a * b + b ** 3
g = grad_forward_2d(f2, [1.0, 2.0])
print(f"Forward grad f=a^2+ab+b^3 в (1,2) = {g}  (теор: (2a+b, a+3b^2) = (4, 13))")

# === Reverse-mode: мини-граф вручную ===
# f(a,b) = (a+b) * (a*b)
# v1=a+b, v2=a*b, y=v1*v2
a_v, b_v = 1.5, -0.7
v1 = a_v + b_v
v2 = a_v * b_v
y = v1 * v2
# обратный проход
dy = 1.0
dv1 = dy * v2
dv2 = dy * v1
# v1=a+b -> da+=dv1, db+=dv1
# v2=a*b -> da+=dv2*b, db+=dv2*a
da = dv1 + dv2 * b_v
db = dv1 + dv2 * a_v
# Сверка с аналитикой: df/da = v2 + v1*b = ab + (a+b)*b = 2ab+b^2
da_an = 2 * a_v * b_v + b_v ** 2
db_an = 2 * a_v * b_v + a_v ** 2
print(f"Reverse AD: da={da:.4f} (an {da_an:.4f}), db={db:.4f} (an {db_an:.4f})")

# === VJP: для F(x) = (x_0^2, x_0*x_1) в x=(1,2), J=[[2,0],[2,1]], v=(3,4)
# v^T J = (3*2+4*2, 3*0+4*1) = (14, 4)
J = np.array([[2.0, 0.0], [2.0, 1.0]])
v = np.array([3.0, 4.0])
print(f"VJP v^T J = {v @ J}  (ожидаемо [14, 4])")
