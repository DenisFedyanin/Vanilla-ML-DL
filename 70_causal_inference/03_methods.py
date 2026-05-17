"""Раздел 70 — Causal inference.

Файл 3: Методы оценки причинных эффектов.

Концепция: Propensity score.
e(x) = P(T=1 | X=x). Сводит многомерный X к одному числу — вероятности лечения.
Если e(x) близко к 0 или 1 — это положительные/отрицательные крайности, оценить
эффект на таких объектах сложно (positivity violated).

Концепция: IPW (Inverse Propensity Weighting).
ATE_IPW = mean(T*Y / e(X)) - mean((1-T)*Y / (1-e(X))). Идея: 'дублируем' редких
treated/untreated, восстанавливая сбалансированную популяцию.

Концепция: Matching (1-NN на propensity score).
Для каждого treated находим ближайший по e(x) контроль, считаем разницу Y и усредняем.
Простая и наглядная техника. Минусы: чувствительность к выбору, потеря несовпавших.

Концепция: Difference-in-Differences (DiD).
Панельные данные до/после интервенции для treated и control групп.
ATE = (Y_treated_after - Y_treated_before) - (Y_control_after - Y_control_before).
Контролирует постоянные конфаундеры (фиксированные эффекты группы и времени).

Концепция: Regression Discontinuity (RD).
Treatment назначается по правилу X >= cutoff. Сравниваем Y чуть выше и чуть ниже cutoff;
все другие переменные меняются плавно, скачок Y у cutoff — каузальный эффект.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(0)

# === симулируем observational с одним X и конфаундингом ===
n = 1500
X = rng.normal(size=(n, 2))
true_e = 1 / (1 + np.exp(-(X @ [1.5, -1.0])))
T = (rng.uniform(size=n) < true_e).astype(int)
true_ate = 2.0
Y = X @ [1.0, 0.5] + true_ate * T + 0.5 * rng.normal(size=n)

# === Propensity score (логистикой) ===
ps = LogisticRegression().fit(X, T).predict_proba(X)[:, 1]
ps = np.clip(ps, 0.05, 0.95)  # обрезка крайностей
print(f"Истинный ATE = {true_ate:.2f}")
print(f"Наивная разность Y(T=1) - Y(T=0) = {Y[T==1].mean() - Y[T==0].mean():.3f}")

# === IPW estimator ===
ate_ipw = (T * Y / ps).mean() - ((1 - T) * Y / (1 - ps)).mean()
print(f"IPW estimator   ATE = {ate_ipw:.3f}")

# === Matching по propensity score (1-NN) ===
idx_t = np.where(T == 1)[0]
idx_c = np.where(T == 0)[0]
ps_t = ps[idx_t]
ps_c = ps[idx_c]
# для каждого treated ищем ближайший control
diffs = np.abs(ps_t[:, None] - ps_c[None, :])
nearest = diffs.argmin(axis=1)
ate_match = (Y[idx_t] - Y[idx_c[nearest]]).mean()
print(f"PS matching     ATE = {ate_match:.3f}")

# === Линейная регрессия с adjustment (для сравнения) ===
XD = np.column_stack([np.ones(n), T, X])
beta, *_ = np.linalg.lstsq(XD, Y, rcond=None)
print(f"Regression adj. ATE = {beta[1]:.3f}")

# === Difference-in-Differences ===
# симулируем 2 группы, 2 момента (до/после), эффект только для treated в after.
G = rng.integers(0, 2, size=200)   # 0 control, 1 treated
# Y_before: общий тренд + group effect
Y_before = 2 * G + rng.normal(size=200)
true_did = 1.5
# Y_after: + общий тренд +0.5 + treated получают +true_did
Y_after = 0.5 + 2 * G + true_did * G + rng.normal(size=200)
did = (Y_after[G == 1].mean() - Y_before[G == 1].mean()) - \
      (Y_after[G == 0].mean() - Y_before[G == 0].mean())
print(f"\nDifference-in-Differences: estimated = {did:.3f}, true = {true_did:.3f}")

# === Regression Discontinuity (концепт + демо) ===
# X_run — переменная, treatment если X_run >= 0. Y зависит от X_run плавно + скачок +tau на cutoff.
X_run = rng.uniform(-1, 1, size=400)
T_rd = (X_run >= 0).astype(int)
true_tau = 1.0
Y_rd = 0.5 * X_run + true_tau * T_rd + 0.2 * rng.normal(size=400)
# берём узкое окно вокруг cutoff и сравниваем средние:
mask = np.abs(X_run) < 0.1
tau_hat = Y_rd[mask & (T_rd == 1)].mean() - Y_rd[mask & (T_rd == 0)].mean()
print(f"Regression Discontinuity: estimated jump at cutoff = {tau_hat:.3f}, true = {true_tau:.3f}")
