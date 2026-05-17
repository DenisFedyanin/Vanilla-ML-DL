"""Раздел 12 — Распределения вероятностей.

Файл 1: основные ДИСКРЕТНЫЕ распределения.

Концепция 120: Bernoulli(p).
Один эксперимент с двумя исходами: 1 (успех) с вероятностью p, 0 — с (1-p).
Среднее p, дисперсия p(1-p). Самый простой случай — подбрасывание монетки.

Концепция 121: Binomial(n, p).
Сколько успехов в n независимых испытаниях Бернулли.
P(X=k) = C(n,k) * p^k * (1-p)^(n-k). E[X]=np, Var=np(1-p).

Концепция 122: Categorical / Multinoulli (k классов).
Обобщение Бернулли на k исходов. Используется в softmax-классификации.

Концепция 123: Multinomial(n, p1..pk).
Сколько раз выпал каждый из k классов за n испытаний.

Концепция 124: Poisson(lambda).
Число редких событий в интервале времени/пространства (звонки в колл-центр,
опечатки в книге). E=Var=lambda. Аппроксимирует Binomial при больших n, малых p.

Концепция 125: Geometric(p).
Сколько испытаний до первого успеха. P(X=k) = (1-p)^(k-1) * p.

Концепция 126: Negative Binomial(r, p).
Сколько неудач до r-го успеха. Обобщение Geometric.

Концепция 127: Hypergeometric(N, K, n).
Сколько 'красных шаров' в выборке n без возвращения из урны (N всего, K красных).

Концепция 128: Discrete Uniform.
Все значения от a до b равновероятны (например, бросок кубика).

Концепция 129: Zipf-распределение.
Частота k-го по популярности слова ~ 1/k^s. Везде в естественном языке.
"""
import numpy as np

rng = np.random.default_rng(0)
N = 100_000

# 120: Bernoulli
b = (rng.uniform(size=N) < 0.3).astype(int)
print(f"120 Bernoulli(0.3): mean={b.mean():.3f}, theor=0.3")

# 121: Binomial(10, 0.3) через сумму Bernoulli
binom = rng.binomial(n=10, p=0.3, size=N)
print(f"121 Binomial(10,0.3): mean={binom.mean():.2f} (=np=3), var={binom.var():.2f} (=np(1-p)=2.1)")

# 122: Categorical (один бросок 6-гранного кубика)
faces = rng.integers(1, 7, size=N)
print(f"122 Categorical: counts per face = {np.bincount(faces)[1:]}")

# 123: Multinomial (5 бросков, 6 граней)
mult = rng.multinomial(n=5, pvals=[1 / 6] * 6, size=5)
print(f"123 Multinomial(5, кубик): 5 экспериментов = \n{mult}")

# 124: Poisson(3)
pois = rng.poisson(lam=3, size=N)
print(f"124 Poisson(3): mean={pois.mean():.2f}, var={pois.var():.2f} (оба =lambda=3)")

# 125: Geometric(0.3)
geom = rng.geometric(p=0.3, size=N)
print(f"125 Geometric(0.3): mean={geom.mean():.2f} (=1/p≈3.33)")

# 126: Negative Binomial (число неудач до 5 успехов)
nb = rng.negative_binomial(n=5, p=0.5, size=N)
print(f"126 NegBin(5,0.5): mean={nb.mean():.2f} (=r(1-p)/p=5)")

# 127: Hypergeometric
hg = rng.hypergeometric(ngood=20, nbad=30, nsample=10, size=N)
print(f"127 Hypergeom(20,30,10): mean={hg.mean():.2f} (=n*K/N=4)")

# 128: Discrete Uniform [1..6]
du = rng.integers(1, 7, size=N)
print(f"128 Uniform discrete 1..6: mean={du.mean():.3f} (теор=3.5)")

# 129: Zipf(2)
zipf = rng.zipf(a=2.0, size=N)
print(f"129 Zipf(s=2): первые 10 рангов = {np.bincount(zipf)[1:11]}")
