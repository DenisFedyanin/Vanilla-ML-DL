"""Раздел 17 — Линейные модели (расширенно).

Файл 4: обобщённые линейные модели (GLM).

Концепция 256: Poisson regression.
Цель — целые неотрицательные счётчики (число звонков, аварий и т.п.).
log E[y|x] = X w. Loss = пуассоновское -log правдоподобие.

Концепция 257: Gamma regression.
Цель — положительные величины с правым хвостом (затраты, время до отказа).
log E[y|x] = X w, отклонение по Gamma-распределению.

Концепция 258: Tweedie regression.
Семейство, обобщающее Normal/Poisson/Gamma через параметр power. Стандарт
в страховании (zero-inflated continuous target).
"""
import numpy as np
from sklearn.linear_model import GammaRegressor, PoissonRegressor, TweedieRegressor

rng = np.random.default_rng(0)
n = 300
X = rng.uniform(0, 2, size=(n, 3))
true_mu = np.exp(0.5 + X @ np.array([1.0, -0.5, 0.3]))
y_pois = rng.poisson(lam=true_mu)
y_gamma = rng.gamma(shape=2.0, scale=true_mu / 2.0)
y_tw = rng.gamma(shape=2.0, scale=true_mu / 2.0) * (rng.uniform(size=n) > 0.3)  # с нулями

# 256
pr = PoissonRegressor(alpha=0.0, max_iter=300).fit(X, y_pois)
print(f"256 Poisson coefs={pr.coef_.round(3)}, intercept={pr.intercept_:.3f} (теор=[1,-0.5,0.3], 0.5)")

# 257
gr = GammaRegressor(alpha=0.0, max_iter=300).fit(X, y_gamma)
print(f"257 Gamma   coefs={gr.coef_.round(3)}, intercept={gr.intercept_:.3f}")

# 258
tw = TweedieRegressor(power=1.5, alpha=0.0, max_iter=300, link="log").fit(X, y_tw)
print(f"258 Tweedie(p=1.5) coefs={tw.coef_.round(3)}, intercept={tw.intercept_:.3f}")
