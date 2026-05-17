"""Раздел 70 — Causal inference.

Файл 4: Uplift modeling.

Концепция: Uplift.
В классическом маркетинге предсказываем 'купит ли клиент'. В uplift предсказываем
ИЗМЕНЕНИЕ вероятности покупки под воздействием treatment (рассылки, скидки):
tau(x) = E[Y(1) - Y(0) | X=x] = CATE (conditional average treatment effect).
Цель: показывать рекламу только тем, у кого tau(x) > 0.

Концепция: Persuadables / Sure-things / Lost-causes / Do-not-disturb.
- Persuadables: купят только при treatment. uplift>0. Цель кампании.
- Sure-things: купят при любом раскладе. uplift~0. Бесполезно.
- Lost-causes: не купят ни при чём. uplift~0. Бесполезно.
- Do-not-disturb (sleeping dogs): treatment ОТТАЛКИВАЕТ. uplift<0. Опасно.

Концепция: S-learner (single model).
Учим одну модель P(Y | X, T), включая T как фичу. uplift(x) = P(Y|x,T=1) - P(Y|x,T=0).
Прост, но T часто 'теряется' среди фич.

Концепция: T-learner (two models).
Учим раздельно mu_1(x) = E[Y|X,T=1] на treated и mu_0(x) = E[Y|X,T=0] на control.
uplift(x) = mu_1(x) - mu_0(x). Сильно зависит от баланса данных в группах.

Концепция: X-learner (Künzel et al.).
Шаг 1: T-learner. Шаг 2: вычисляем 'импутированные' эффекты:
  D^1_i = Y_i - mu_0(X_i) для treated; D^0_i = mu_1(X_i) - Y_i для control.
Шаг 3: учим tau_1(x) на (X_treated, D^1) и tau_0(x) на (X_control, D^0).
Финал: tau(x) = g(x) * tau_0(x) + (1-g(x)) * tau_1(x), где g(x) — propensity score.

Концепция: Qini curve.
Похоже на gain curve в classification. Сортируем по предсказанному uplift desc,
для каждого top-k% считаем реальный uplift на этой подгруппе. Площадь под Qini —
качество uplift-модели.

Демо: на симулированных A/B-данных строим T-learner и считаем gain.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(0)

# === симулируем A/B-эксперимент с разной чувствительностью ===
n = 4000
X = rng.normal(size=(n, 3))
T = rng.integers(0, 2, size=n)  # рандом — это RCT
# истинный uplift зависит от X[:,0]: чем больше — тем сильнее эффект
true_uplift = 0.4 * X[:, 0]
prob_y = 1 / (1 + np.exp(-(0.5 * X[:, 1] + true_uplift * T)))
Y = (rng.uniform(size=n) < prob_y).astype(int)

# === T-learner ===
m1 = LogisticRegression().fit(X[T == 1], Y[T == 1])
m0 = LogisticRegression().fit(X[T == 0], Y[T == 0])
p1 = m1.predict_proba(X)[:, 1]
p0 = m0.predict_proba(X)[:, 1]
uplift_t = p1 - p0

# === S-learner ===
XS = np.column_stack([X, T])
ms = LogisticRegression().fit(XS, Y)
p1_s = ms.predict_proba(np.column_stack([X, np.ones(n)]))[:, 1]
p0_s = ms.predict_proba(np.column_stack([X, np.zeros(n)]))[:, 1]
uplift_s = p1_s - p0_s

# === качество: корреляция с истинным uplift ===
cor_t = np.corrcoef(uplift_t, true_uplift)[0, 1]
cor_s = np.corrcoef(uplift_s, true_uplift)[0, 1]
print(f"Correlation predicted uplift vs true uplift:")
print(f"  T-learner: {cor_t:.3f}")
print(f"  S-learner: {cor_s:.3f}")

# === Qini-подобная кривая ===
order = np.argsort(-uplift_t)
n_treat = np.cumsum(T[order])
n_ctrl = np.cumsum(1 - T[order])
# средний Y в treated и control в топ-k:
def cum_mean(arr, mask_order):
    cs = np.cumsum(arr[mask_order])
    return cs

cs_yt = np.cumsum(Y[order] * T[order])
cs_yc = np.cumsum(Y[order] * (1 - T[order]))
# uplift в каждом top-k: rate_treated - rate_control
with np.errstate(invalid="ignore", divide="ignore"):
    rate_t = cs_yt / np.maximum(n_treat, 1)
    rate_c = cs_yc / np.maximum(n_ctrl, 1)
qini_curve = (rate_t - rate_c) * np.arange(1, n + 1)  # incremental gain
auc_qini = np.trapezoid(qini_curve, dx=1) / n

# для random ordering:
order_r = rng.permutation(n)
cs_yt_r = np.cumsum(Y[order_r] * T[order_r])
cs_yc_r = np.cumsum(Y[order_r] * (1 - T[order_r]))
n_treat_r = np.cumsum(T[order_r])
n_ctrl_r = np.cumsum(1 - T[order_r])
with np.errstate(invalid="ignore", divide="ignore"):
    rate_t_r = cs_yt_r / np.maximum(n_treat_r, 1)
    rate_c_r = cs_yc_r / np.maximum(n_ctrl_r, 1)
qini_rand = (rate_t_r - rate_c_r) * np.arange(1, n + 1)
auc_qini_r = np.trapezoid(qini_rand, dx=1) / n

print(f"\nQini AUC (T-learner): {auc_qini:.2f}")
print(f"Qini AUC (random)   : {auc_qini_r:.2f}")
print("Положительная Qini AUC сверх random — модель действительно ранжирует persuadables первыми.")

# === X-learner (концепт + демо) ===
# импутируем эффекты
D1 = Y[T == 1] - m0.predict_proba(X[T == 1])[:, 1]
D0 = m1.predict_proba(X[T == 0])[:, 1] - Y[T == 0]
# по-простому: усреднение двух моделей предсказания uplift
from sklearn.linear_model import LinearRegression
tau1 = LinearRegression().fit(X[T == 1], D1).predict(X)
tau0 = LinearRegression().fit(X[T == 0], D0).predict(X)
g = T.mean()
uplift_x = g * tau0 + (1 - g) * tau1
cor_x = np.corrcoef(uplift_x, true_uplift)[0, 1]
print(f"\nX-learner correlation with true uplift: {cor_x:.3f}")
