"""Раздел 29 — Активации (расширенно). Файл 1: семейство ReLU и гладкие.

Концепция 301: ReLU(x) = max(0, x).
Простая, быстрая, не насыщается при x>0. Минус — "мертвые нейроны" при x<0.

Концепция 302: LeakyReLU(x) = x if x>0 else alpha*x.
Маленький наклон в отрицательной зоне (alpha=0.01) — нет мертвых нейронов.

Концепция 303: PReLU.
То же что LeakyReLU, но alpha — обучаемый параметр.

Концепция 304: ELU(x) = x if x>0 else alpha*(exp(x)-1).
Гладкая, отрицательные значения насыщаются к -alpha. Среднее активаций ближе к нулю.

Концепция 305: CELU.
ELU с параметром alpha, нормированным как exp(x/alpha)-1.

Концепция 306: SELU = lambda * ELU(alpha).
Self-Normalizing: с правильной инициализацией поддерживает mean=0, var=1.
Константы фиксированы: lambda≈1.0507, alpha≈1.6733.

Концепция 307: GELU(x) ≈ x * Phi(x).
Гладкая ReLU-подобная. Используется в BERT/GPT. Точная: x*Phi(x);
приближённая: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3))).

Концепция 308: Swish / SiLU(x) = x * sigmoid(x).
Гладкая, без верхнего предела, с лёгким провалом — улучшает обучение.

Концепция 309: Mish(x) = x * tanh(softplus(x)).
Похожа на Swish, чуть мягче, популярна в YOLO.

Концепция 310: Softplus(x) = log(1+exp(x)).
Гладкое приближение ReLU. Всегда положительна.

Концепция 311: Softsign(x) = x / (1+|x|).
Альтернатива tanh, медленнее насыщается.

Концепция 312: HardSigmoid / HardTanh / HardSwish.
Кусочно-линейные приближения для эффективных вычислений (mobile).
"""
import numpy as np

x = np.linspace(-3, 3, 7)

def relu(x): return np.maximum(0, x)
def leaky(x, a=0.01): return np.where(x > 0, x, a * x)
def prelu(x, a): return np.where(x > 0, x, a * x)
def elu(x, a=1.0): return np.where(x > 0, x, a * (np.exp(x) - 1))
def celu(x, a=1.0): return np.where(x > 0, x, a * (np.exp(x / a) - 1))
def selu(x):
    lam, a = 1.0507, 1.6733
    return lam * np.where(x > 0, x, a * (np.exp(x) - 1))
def gelu_exact(x):
    # exact: x * Phi(x), где Phi — CDF стандартной нормальной
    return 0.5 * x * (1 + np.vectorize(_erf)(x / np.sqrt(2)))
def _erf(z):
    # численная аппроксимация erf без scipy (Abramowitz & Stegun 7.1.26)
    t = 1.0 / (1.0 + 0.3275911 * abs(z))
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-z * z)
    return np.sign(z) * y
def gelu_approx(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
def swish(x): return x / (1 + np.exp(-x))
def mish(x): return x * np.tanh(np.log1p(np.exp(x)))
def softplus(x): return np.log1p(np.exp(x))
def softsign(x): return x / (1 + np.abs(x))
def hard_sigmoid(x): return np.clip(0.2 * x + 0.5, 0, 1)
def hard_tanh(x): return np.clip(x, -1, 1)
def hard_swish(x): return x * np.clip((x + 3) / 6, 0, 1)

print(f"x = {x.tolist()}")
print(f"301 ReLU      = {relu(x).round(3).tolist()}")
print(f"302 LeakyReLU = {leaky(x).round(3).tolist()}")
print(f"303 PReLU a=0.25 = {prelu(x, 0.25).round(3).tolist()}")
print(f"304 ELU       = {elu(x).round(3).tolist()}")
print(f"305 CELU a=0.5= {celu(x, 0.5).round(3).tolist()}")
print(f"306 SELU      = {selu(x).round(3).tolist()}")
print(f"307 GELU exact= {gelu_exact(x).round(3).tolist()}")
print(f"    GELU appr = {gelu_approx(x).round(3).tolist()}  (близко к exact)")
print(f"308 Swish     = {swish(x).round(3).tolist()}")
print(f"309 Mish      = {mish(x).round(3).tolist()}")
print(f"310 Softplus  = {softplus(x).round(3).tolist()}")
print(f"311 Softsign  = {softsign(x).round(3).tolist()}")
print(f"312 HardSigmoid={hard_sigmoid(x).round(3).tolist()}")
print(f"    HardTanh  = {hard_tanh(x).round(3).tolist()}")
print(f"    HardSwish = {hard_swish(x).round(3).tolist()}")
