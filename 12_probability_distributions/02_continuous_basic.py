"""Раздел 12 — Распределения вероятностей.

Файл 2: основные НЕПРЕРЫВНЫЕ распределения.

Концепция 130: Uniform(a, b).
Все значения отрезка [a,b] равновероятны. Базовый кирпич для остальных
генераторов (inverse-CDF, rejection sampling). E=(a+b)/2, Var=(b-a)^2/12.

Концепция 131: Normal/Gaussian(mu, sigma^2).
Колоколообразная кривая. ЦПТ: сумма независимых одинаково распределённых
случайных величин стремится к нормальному. Базис почти всей статистики.

Концепция 132: Exponential(lambda).
Время до следующего "редкого" события (Пуассон-процесс). Без памяти:
P(X>s+t|X>s)=P(X>t). E=1/lambda, Var=1/lambda^2.

Концепция 133: Laplace(mu, b).
Двусторонняя экспонента: pdf ~ exp(-|x-mu|/b). Тяжелее хвосты, чем у нормали.
Используется в L1-регуляризации (априор Лапласа = Lasso).

Концепция 134: Cauchy(x0, gamma).
Очень тяжёлые хвосты — у Cauchy нет матожидания и дисперсии. Часто
используется как контрпример к ЦПТ. pdf ~ 1/(1+((x-x0)/gamma)^2).

Концепция 135: LogNormal(mu, sigma).
Если log(X) ~ Normal, то X ~ LogNormal. Положительная случайная величина с
длинным правым хвостом: зарплаты, размеры файлов, цены.

Концепция 136: Pareto(alpha, xm).
Степенной хвост: P(X>x) = (xm/x)^alpha. "Правило 80/20": 20% людей
владеют 80% богатства. Дисперсии может не быть.

Концепция 137: Weibull(k, lambda).
Распределение времени до отказа в теории надёжности. Параметр формы k
определяет, "стареет" ли система (k>1) или нет (k=1 — экспонента).

Концепция 138: Gumbel.
Распределение максимумов выборок (Extreme Value Theory тип I). Появляется
в Gumbel-softmax trick для дифференцируемого сэмплинга категориальных.

Концепция 139: Rayleigh(sigma).
Модуль 2D-нормального вектора с независимыми компонентами. Скорости частиц,
ошибки радара.
"""
import numpy as np

rng = np.random.default_rng(0)
N = 100_000

# 130: Uniform(0,1)
u = rng.uniform(0, 1, size=N)
print(f"130 Uniform(0,1): mean={u.mean():.3f} (теор=0.5), var={u.var():.3f} (теор≈0.083)")

# 131: Normal(2, 0.5)
g = rng.normal(loc=2.0, scale=0.5, size=N)
print(f"131 Normal(2, 0.5): mean={g.mean():.3f}, std={g.std():.3f}")

# 132: Exponential(lambda=2) => scale=1/lambda=0.5
e = rng.exponential(scale=0.5, size=N)
print(f"132 Exponential(lam=2): mean={e.mean():.3f} (теор=0.5)")

# 133: Laplace(0, 1)
lap = rng.laplace(loc=0.0, scale=1.0, size=N)
print(f"133 Laplace(0,1): mean={lap.mean():.3f}, var={lap.var():.3f} (теор=2)")

# 134: Cauchy(0,1) — медиана есть, среднего нет
cau = rng.standard_cauchy(size=N)
print(f"134 Cauchy: median={np.median(cau):.3f}, mean нестабильно ({cau.mean():.1f})")

# 135: LogNormal — exp от нормального
ln = rng.lognormal(mean=0.0, sigma=1.0, size=N)
print(f"135 LogNormal(0,1): median={np.median(ln):.3f} (=exp(0)=1), mean={ln.mean():.2f}")

# 136: Pareto (numpy даёт сдвиг: реальное X = (1+pareto)*xm)
par = (rng.pareto(a=3.0, size=N) + 1)  # xm=1
print(f"136 Pareto(alpha=3, xm=1): median={np.median(par):.3f}, mean={par.mean():.3f}")

# 137: Weibull(k=2)
w = rng.weibull(a=2.0, size=N)
print(f"137 Weibull(k=2, lam=1): mean={w.mean():.3f} (теор≈0.886)")

# 138: Gumbel — sampling трюк: argmax(logits + Gumbel) ~ categorical
gu = rng.gumbel(loc=0.0, scale=1.0, size=N)
print(f"138 Gumbel(0,1): mean={gu.mean():.3f} (теор≈0.577 — Эйлер-Маскерони)")

# 139: Rayleigh = sqrt(X^2+Y^2), X,Y ~ N(0,sigma)
x = rng.normal(0, 1.0, size=N)
y = rng.normal(0, 1.0, size=N)
ray = np.sqrt(x ** 2 + y ** 2)
print(f"139 Rayleigh(sigma=1): mean={ray.mean():.3f} (теор=sigma*sqrt(pi/2)≈1.253)")
