"""Раздел 61 — Вероятность.

Файл 2: Моменты случайных величин.

Концепция: Линейность матожидания.
E[aX + bY] = a E[X] + b E[Y]. Работает ВСЕГДА, даже для зависимых X, Y.

Концепция: Дисперсия.
Var(X) = E[(X-mu)^2] = E[X^2] - mu^2.
Var(aX+b) = a^2 Var(X). Для независимых: Var(X+Y) = Var(X)+Var(Y).

Концепция: Ковариация и корреляция.
Cov(X,Y) = E[(X-mu_X)(Y-mu_Y)]. Corr = Cov / (sigma_X * sigma_Y) in [-1,1].
Независимость => Cov=0, но обратное неверно (только линейная связь).

Концепция: Высшие моменты.
m_k = E[X^k]; центральные мю_k = E[(X-mu)^k].
Асимметрия (skew) = мю_3 / sigma^3, эксцесс (kurtosis) = мю_4 / sigma^4 (норм. имеет 3).

Концепция: Производящая функция моментов (MGF).
M_X(t) = E[e^{tX}]. Если существует в окрестности 0, m_k = M^{(k)}(0).

Концепция: Характеристическая функция.
phi_X(t) = E[e^{itX}]. Существует всегда. Преобразование Фурье плотности.
"""
import numpy as np

rng = np.random.default_rng(0)
N = 100_000

X = rng.normal(loc=2.0, scale=1.0, size=N)
Y = 0.7 * X + 0.3 * rng.normal(size=N)  # коррелирован с X

# === Линейность ===
print(f"E[3X-2Y]={3*X.mean()-2*Y.mean():.3f} vs E[3X-2Y]={(3*X-2*Y).mean():.3f}")

# === Дисперсия ===
print(f"Var(X)={X.var():.4f}, Var(2X+5)={(2*X+5).var():.4f} (теор 4*Var={4*X.var():.4f})")

# === Ковариация и корреляция ===
cov = ((X - X.mean()) * (Y - Y.mean())).mean()
corr = cov / (X.std() * Y.std())
print(f"Cov(X,Y)={cov:.4f}, Corr={corr:.4f}")

# Контрпример: зависимые но Cov=0
U = rng.normal(size=N); V = U ** 2 - 1
cov_uv = ((U - U.mean()) * (V - V.mean())).mean()
print(f"V=U^2-1: Cov={cov_uv:.4f} (~0), но V зависит от U")

# === Высшие моменты ===
mu = X.mean(); s = X.std()
skew = ((X - mu) ** 3).mean() / s ** 3
kurt = ((X - mu) ** 4).mean() / s ** 4
print(f"Skew(X)={skew:.4f}  (~0 для норм.), Kurt={kurt:.4f}  (~3 для норм.)")

# === MGF для N(mu, sigma^2): M(t) = exp(mu t + sigma^2 t^2 / 2) ===
t = 0.5
M_emp = np.mean(np.exp(t * X))
M_theor = np.exp(2.0 * t + 0.5 * t ** 2)  # mu=2, sigma=1
print(f"MGF(0.5): emp={M_emp:.4f}, theor={M_theor:.4f}")

# === Характеристическая функция: phi(t) для N(mu,sigma^2) = exp(i mu t - sigma^2 t^2/2)
phi_emp = np.mean(np.exp(1j * t * X))
phi_theor = np.exp(1j * 2.0 * t - 0.5 * t ** 2)
print(f"phi(0.5): emp={phi_emp:.4f}, theor={phi_theor:.4f}")
