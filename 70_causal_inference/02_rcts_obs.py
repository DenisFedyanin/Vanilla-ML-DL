"""Раздел 70 — Causal inference.

Файл 2: RCT, observational studies, Simpson's paradox.

Концепция: RCT (Randomized Controlled Trial).
Treatment T назначается СЛУЧАЙНО, независимо от X. Тогда T ⫫ конфаундеры,
и
  ATE = E[Y | T=1] - E[Y | T=0]
является несмещённой оценкой среднего причинного эффекта.

Концепция: ATE (Average Treatment Effect).
ATE = E[Y(1) - Y(0)], где Y(t) — потенциальный исход при treatment=t. В RCT
оценивается простой разностью средних.

Концепция: Observational studies.
T назначается НЕ случайно, обычно зависит от X (например, врач выписывает
лекарство тяжёлым пациентам). Тогда сравнение средних смещено конфаундерами.

Концепция: Simpson's paradox.
Эффект, наблюдаемый в общем (агрегированном) виде, может ОБРАТИТЬСЯ при
расщеплении по подгруппам (и наоборот). Возникает из-за неравномерного
распределения конфаундера по группам treatment.

Демо: симулируем 'наблюдательные' данные с тяжестью болезни X в качестве
конфаундера. Простая разность средних даёт неправильный знак; RCT-симуляция
показывает истинный положительный эффект.
"""
import numpy as np

rng = np.random.default_rng(0)
n = 2000

# === Сценарий: лечение T снижает риск Y, но врач даёт его тяжёлым пациентам (X высокое) ===
# X = тяжесть (positive), T более вероятен при больших X, Y = X - 1*T + шум
X = rng.normal(0, 1, size=n)
# обсервационный T
T_obs = (rng.uniform(size=n) < 1 / (1 + np.exp(-2 * X))).astype(int)
true_effect = -1.0   # лечение УМЕНЬШАЕТ Y
Y_obs = 2 * X + true_effect * T_obs + 0.5 * rng.normal(size=n)

ate_naive = Y_obs[T_obs == 1].mean() - Y_obs[T_obs == 0].mean()
print(f"=== Observational study ===")
print(f"  Наивная разность средних Y: {ate_naive:+.3f}")
print(f"  Истинный эффект T -> Y    : {true_effect:+.3f}")
print(f"  Знак неправильный — конфаундер X смещает оценку.")

# === RCT: T случайно ===
T_rct = (rng.uniform(size=n) < 0.5).astype(int)
Y_rct = 2 * X + true_effect * T_rct + 0.5 * rng.normal(size=n)
ate_rct = Y_rct[T_rct == 1].mean() - Y_rct[T_rct == 0].mean()
print(f"\n=== RCT (random T) ===")
print(f"  Разность средних Y       : {ate_rct:+.3f}  (~= истине)")

# === Корректировка через регрессию в наблюдательных данных ===
# Y ~ T + X -> коэффициент при T ~ истинному эффекту
XD = np.column_stack([np.ones(n), T_obs, X])
beta, *_ = np.linalg.lstsq(XD, Y_obs, rcond=None)
print(f"\n=== Linear regression Y ~ T + X на observational ===")
print(f"  coef T (estimated ATE)   : {beta[1]:+.3f}")

# === Simpson's paradox ===
# Лекарство 'работает' внутри каждой подгруппы пациентов, но в агрегате — наоборот
# из-за неравномерного назначения.
print(f"\n=== Simpson's paradox demo ===")
# 2 подгруппы (лёгкие/тяжёлые), эффект лечения положительный в обеих,
# но лекарство дают в основном тяжёлым (у которых Y хуже в среднем).
grp = rng.integers(0, 2, size=n)            # 0=лёгкие, 1=тяжёлые
# вероятность T: 0.2 для лёгких, 0.8 для тяжёлых
p_t = np.where(grp == 0, 0.2, 0.8)
T = (rng.uniform(size=n) < p_t).astype(int)
Y = 0.2 * T + 0.0 * grp - 1.0 * grp + 0.2 * rng.normal(size=n)
# Заметим: лечение даёт +0.2, тяжёлый статус даёт -1.0.
ate_agg = Y[T == 1].mean() - Y[T == 0].mean()
ate_g0 = Y[(T == 1) & (grp == 0)].mean() - Y[(T == 0) & (grp == 0)].mean()
ate_g1 = Y[(T == 1) & (grp == 1)].mean() - Y[(T == 0) & (grp == 1)].mean()
print(f"  ATE в подгруппе 0 (лёгкие)  : {ate_g0:+.3f}")
print(f"  ATE в подгруппе 1 (тяжёлые) : {ate_g1:+.3f}")
print(f"  Агрегированный 'ATE'        : {ate_agg:+.3f}  (может иметь обратный знак)")
print("Внутри подгрупп лечение работает; в агрегате — парадокс Симпсона.")
