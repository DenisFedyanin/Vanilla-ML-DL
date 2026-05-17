"""Раздел 57 — HPO.

Файл 3: TPE, CMA-ES, PBT.

Концепция: TPE (Tree-structured Parzen Estimator).
Разделяет наблюдения по квантилю gamma на "хорошие" (l(x)) и "плохие" (g(x)).
Моделирует l, g непараметрически (KDE). Точка выбирается чтобы максимизировать l(x)/g(x)
— это пропорционально EI без GP. Хорошо работает с категориальными.

Концепция: CMA-ES (Covariance Matrix Adaptation).
Эволюционная стратегия: на каждой итерации sample N точек из N(m, sigma^2 C),
выбираем лучших mu, обновляем среднее m, ковариацию C и шаг sigma.
Не требует градиента, хорошо для blackbox в 5-50 измерениях.

Концепция: PBT (Population-Based Training).
Параллельно тренируется популяция моделей с разными гиперпараметрами.
Периодически слабые "exploit" — копируют веса+гиперы лучших, потом "explore" — пермутируют гиперы.
Адаптирует расписание гиперов прямо во время обучения.
"""
import numpy as np

rng = np.random.default_rng(0)

# === TPE: маленький ручной демо ===
# Целевая f(x) (одномерная, минимизируем)
f = lambda x: (x - 0.7) ** 2 + 0.1 * np.sin(20 * x)
X = rng.uniform(0, 1, size=20)
y = f(X)
gamma = 0.25
thresh = np.quantile(y, gamma)
good = X[y <= thresh]; bad = X[y > thresh]

# KDE через гауссовы ядра, ширина по правилу Сильвермана
def kde(samples, x, h=None):
    h = h or 1.06 * samples.std() * len(samples) ** (-1 / 5) + 1e-3
    diffs = (x[:, None] - samples[None, :]) / h
    return np.mean(np.exp(-0.5 * diffs ** 2) / (np.sqrt(2 * np.pi) * h), axis=1)

grid = np.linspace(0, 1, 200)
ratio = kde(good, grid) / (kde(bad, grid) + 1e-9)
x_next = grid[np.argmax(ratio)]
print(f"TPE: предложенный x={x_next:.3f}, истинный аргмин ~0.7")

# === CMA-ES на 2D-функции (упрощенный) ===
target = lambda x: (x[0] - 1) ** 2 + 4 * (x[1] + 0.5) ** 2
m = np.array([3.0, 3.0]); sigma = 1.0
C = np.eye(2); pop = 12; mu_sel = 4
weights = np.log(mu_sel + 1) - np.log(np.arange(1, mu_sel + 1))
weights /= weights.sum()
for _ in range(30):
    L = np.linalg.cholesky(C)
    samples = m + sigma * (L @ rng.normal(size=(2, pop))).T
    vals = np.array([target(s) for s in samples])
    order = np.argsort(vals)[:mu_sel]
    selected = samples[order]
    m_new = weights @ selected
    diffs = selected - m
    C = sum(w * np.outer(d, d) for w, d in zip(weights, diffs)) / sigma ** 2
    sigma *= np.exp(0.1 * (np.linalg.norm(m_new - m) / sigma - 1))
    m = m_new
print(f"CMA-ES: m={m.round(3)} (оптимум [1, -0.5]), sigma={sigma:.3f}")

# === PBT: 4 агента, гипер = lr ===
N = 4
lrs = rng.uniform(1e-3, 1.0, size=N)
losses = rng.uniform(1, 3, size=N)  # "стартовые" значения
for step in range(8):
    # шаг "обучения": loss падает быстрее при оптимальном lr~0.1
    losses -= 0.1 * np.exp(-((lrs - 0.1) ** 2) / 0.05)
    # exploit/explore: нижние 25% копируют верхние 25%
    order = np.argsort(losses)
    top, bot = order[:N // 4], order[-N // 4:]
    for b, t in zip(bot, top):
        lrs[b] = lrs[t] * rng.choice([0.8, 1.25])
        losses[b] = losses[t]
print(f"PBT: финальные lrs={lrs.round(3)}, losses={losses.round(3)}")
