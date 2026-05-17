"""Раздел 15 — Feature Engineering.

Файл 1: преобразования числовых признаков.

Концепция 198: Log-преобразование.
log(1+x) сжимает длинный правый хвост. Полезно для зарплат, размеров файлов,
цен. log1p корректно работает с нулями.

Концепция 199: Sqrt.
Мягче, чем log. Хорошо для счётчиков (Пуассон-подобных переменных).

Концепция 200: Обратное (reciprocal).
1/x усиливает различия маленьких значений. Применяется редко, при особо
длинных хвостах.

Концепция 201: Box-Cox.
Параметрическое семейство: (x^lambda - 1)/lambda при lambda!=0, log(x)
при lambda=0. Подбирает lambda максимизацией нормальности (через
правдоподобие). Работает только для x>0.

Концепция 202: Yeo-Johnson.
Обобщение Box-Cox на любые x (включая отрицательные). 4 ветки в зависимости
от знака x и значения lambda.

Концепция 203: Равномерное (equal-width) биннинг.
Разбиваем [min,max] на k равных интервалов. Простой и быстрый.

Концепция 204: Квантильный (equal-frequency) биннинг.
Разбиваем по квантилям, в каждом бине ~одинаковое число точек. Робастен
к выбросам, нелинейные эффекты по разные стороны медианы.

Концепция 205: Дискретизация в ординальное.
Превращаем бины в порядковые номера 0..k-1. Часто как вход для tree-based
моделей или для weight-of-evidence.
"""
import numpy as np

rng = np.random.default_rng(0)
x_pos = rng.lognormal(mean=0, sigma=1.0, size=10)  # положительные с правым хвостом
x_any = rng.normal(loc=0, scale=2, size=10)  # любых знаков

# 198: log1p
print(f"198 log1p: {np.log1p(x_pos).round(3)}")

# 199: sqrt
print(f"199 sqrt:  {np.sqrt(x_pos).round(3)}")

# 200: reciprocal
print(f"200 1/x:   {(1.0 / x_pos).round(3)}")

# 201: Box-Cox с поиском lambda
def boxcox(x, lam):
    return np.log(x) if lam == 0 else (x ** lam - 1) / lam


def boxcox_loglik(x, lam):
    t = boxcox(x, lam)
    n = len(x)
    var = t.var()
    return -n / 2 * np.log(var) + (lam - 1) * np.log(x).sum()


lambdas = np.linspace(-2, 2, 41)
best_lam = max(lambdas, key=lambda lm: boxcox_loglik(x_pos, lm))
print(f"201 Box-Cox best lambda={best_lam:.2f}, transformed[:5]={boxcox(x_pos, best_lam)[:5].round(3)}")

# 202: Yeo-Johnson
def yeojohnson(x, lam):
    out = np.zeros_like(x, dtype=float)
    pos = x >= 0
    if lam != 0:
        out[pos] = ((x[pos] + 1) ** lam - 1) / lam
    else:
        out[pos] = np.log(x[pos] + 1)
    if lam != 2:
        out[~pos] = -(((-x[~pos] + 1) ** (2 - lam) - 1) / (2 - lam))
    else:
        out[~pos] = -np.log(-x[~pos] + 1)
    return out


print(f"202 Yeo-Johnson lam=0.5: {yeojohnson(x_any, 0.5).round(3)}")

# 203: equal-width binning
k = 4
edges_ew = np.linspace(x_any.min(), x_any.max(), k + 1)
bins_ew = np.clip(np.digitize(x_any, edges_ew[1:-1]), 0, k - 1)
print(f"203 equal-width bins (k={k}): {bins_ew}")

# 204: quantile binning
qs = np.linspace(0, 1, k + 1)
edges_q = np.quantile(x_any, qs)
bins_q = np.clip(np.digitize(x_any, edges_q[1:-1]), 0, k - 1)
print(f"204 quantile bins: {bins_q}")

# 205: ordinal — те же бины уже являются ординальными метками
print(f"205 ordinal labels (по quantile): {bins_q.tolist()}")
