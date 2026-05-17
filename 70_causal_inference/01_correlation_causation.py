"""Раздел 70 — Causal inference.

Файл 1: Correlation vs causation, confounders, partial correlation.

Концепция: Correlation != causation.
Корреляция между X и Y НЕ означает, что X вызывает Y. Возможны: X->Y, Y->X,
общий предок Z->X, Z->Y (конфаундер), случайность, отбор выборки.

Концепция: Confounder.
Переменная Z, влияющая И на X, И на Y. Создаёт ложную корреляцию между X и Y.
Пример: продажи мороженого X и утопления Y коррелируют, но общая причина —
жаркая погода Z.

Концепция: Partial correlation.
Корреляция X и Y, очищенная от влияния Z: считаем остатки X | Z и Y | Z
(после линейной регрессии на Z) и берём cor этих остатков.
Если cor(X, Y) высокая, а partial cor(X, Y | Z) ~ 0 — Z был конфаундером.

Концепция: Backdoor adjustment.
Чтобы оценить причинный эффект X -> Y, нужно 'закрыть' все backdoor-пути
(X <- Z -> Y), включив Z в анализ (в линейном случае — в регрессию).

Демо: симулируем Z -> X, Z -> Y. cor(X, Y) высокая, partial cor ~ 0.
"""
import numpy as np

rng = np.random.default_rng(0)

# === Z -> X, Z -> Y (нет прямой связи X -> Y) ===
n = 1000
Z = rng.normal(size=n)
X = 1.5 * Z + 0.5 * rng.normal(size=n)
Y = 2.0 * Z + 0.5 * rng.normal(size=n)

cor_xy = np.corrcoef(X, Y)[0, 1]
print(f"cor(X, Y) = {cor_xy:.3f}  (высокая, хотя X не влияет на Y!)")

# === partial cor(X, Y | Z): остатки линейной регрессии ===
def residuals(target, predictors):
    Pred = np.column_stack([np.ones(len(predictors)), predictors])
    beta, *_ = np.linalg.lstsq(Pred, target, rcond=None)
    return target - Pred @ beta

resX = residuals(X, Z)
resY = residuals(Y, Z)
pcor = np.corrcoef(resX, resY)[0, 1]
print(f"partial cor(X, Y | Z) = {pcor:.3f}  (~0, конфаундинг разоблачён)")

# === сравним с настоящей причинной связью X -> Y ===
print("\n--- Сценарий 2: настоящая X -> Y (без конфаундера) ---")
X2 = rng.normal(size=n)
Y2 = 1.0 * X2 + 0.5 * rng.normal(size=n)
Z2 = rng.normal(size=n)
print(f"cor(X, Y)            = {np.corrcoef(X2, Y2)[0, 1]:.3f}")
print(f"partial cor(X, Y|Z)  = "
      f"{np.corrcoef(residuals(X2, Z2), residuals(Y2, Z2))[0, 1]:.3f}  (всё ещё высока)")

# === Backdoor adjustment в линейной регрессии ===
# Если регрессировать Y на X без Z, коэффициент при X будет смещён.
def lin_coef(target, X_design):
    XD = np.column_stack([np.ones(len(X_design)), X_design])
    beta, *_ = np.linalg.lstsq(XD, target, rcond=None)
    return beta[1:]

print("\n--- Сценарий 1 (есть конфаундер Z), регрессии Y ~ X: ---")
b_naive = lin_coef(Y, X)
print(f"  без Z: coef X = {b_naive[0]:.3f}  (далеко от 0, ложно!)")
b_adj = lin_coef(Y, np.column_stack([X, Z]))
print(f"  с Z  : coef X = {b_adj[0]:.3f}, coef Z = {b_adj[1]:.3f}  (X близко к 0)")

print("\nИтог: чтобы оценить КАУЗАЛЬНЫЙ эффект, нужно контролировать конфаундеры.")
