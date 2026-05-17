"""Раздел 55 — MCMC и сэмплирование. Файл 1: Простые методы.

Концепция: Inverse CDF sampling.
Если CDF F можно инвертировать, то X = F^{-1}(U), U ~ Uniform(0,1), имеет
распределение с CDF F. Работает только для распределений с явной F^{-1}.
Пример: Exp(lambda) -> F^{-1}(u) = -ln(1 - u) / lambda.

Концепция: Rejection sampling.
Хотим сэмплировать из p(x). Берём пропозицию q(x), такую что M*q(x) >= p(x).
Сэмплируем x ~ q, принимаем с вероятностью p(x)/(M*q(x)). Эффективность —
1/M. Чем хуже q приближает p, тем больше отказов.

Концепция: Importance sampling.
Не сэмплируем из p, а оцениваем E_p[f(X)] = ∫ f(x) p(x) dx
                                          = ∫ f(x) (p(x)/q(x)) q(x) dx
                                          ≈ (1/N) sum f(x_i) * w(x_i), x_i ~ q.
w_i = p(x_i)/q(x_i) — importance weights. Полезно для редких событий.

Концепция: Effective sample size (ESS).
ESS = (sum w)^2 / sum w^2. Если веса сильно неравномерны (один w доминирует),
ESS << N — эффективных образцов мало. Признак плохого выбора q.

Концепция: Когда что использовать.
Inverse CDF — простой, но требует F^{-1}. Rejection — годится для маленьких
размерностей. Importance — для оценки интегралов / редких событий.
В высоких размерностях все три страдают от проклятия размерности; нужны MCMC.
"""
import numpy as np

rng = np.random.default_rng(0)
N = 20000

# === Inverse CDF: Exp(lambda=2) ===
lam = 2.0
u = rng.uniform(size=N)
x_exp = -np.log(1 - u) / lam
print(f"Exp(2) inv-CDF: mean={x_exp.mean():.3f}  теор=1/lambda={1/lam}")

# === Rejection sampling: target = mixture N(-2,1) + N(2,1), proposal = N(0, 3) ===
def target_pdf(x):
    return 0.5 * np.exp(-0.5 * (x + 2) ** 2) / np.sqrt(2 * np.pi) + \
           0.5 * np.exp(-0.5 * (x - 2) ** 2) / np.sqrt(2 * np.pi)


def proposal_pdf(x, s=3.0):
    return np.exp(-0.5 * (x / s) ** 2) / (s * np.sqrt(2 * np.pi))


M = 6.0    # верхняя оценка p(x)/q(x)
samples = []
attempts = 0
while len(samples) < N:
    attempts += 1
    z = rng.normal(scale=3.0)
    if rng.uniform() < target_pdf(z) / (M * proposal_pdf(z)):
        samples.append(z)
samples = np.array(samples)
print(f"Rejection: acceptance rate = {N / attempts:.3f}  (теор ~ 1/M)")
print(f"  mean={samples.mean():+.3f} (теор=0), std≈{samples.std():.2f}")

# === Importance sampling: оценим E_p[X^2] ===
N_is = 5000
z = rng.normal(scale=3.0, size=N_is)
w = target_pdf(z) / proposal_pdf(z)
est = (w * z ** 2).sum() / w.sum()
ess = (w.sum() ** 2) / (w ** 2).sum()
true_E = 0.5 * (1 + 4) + 0.5 * (1 + 4)  # E[X^2] = sigma^2 + mu^2 for each mixture
print(f"Importance E[X^2] ≈ {est:.3f}  (теор={true_E})  ESS={ess:.0f}/{N_is}")
