"""Раздел 50 — Interpretability. Файл 3: Локальные объяснения (LIME / SHAP).

Концепция: Локальная интерпретация.
'Почему модель решила, что этот клиент — мошенник?' Не глобальная важность,
а вклад каждой фичи в КОНКРЕТНОЕ предсказание.

Концепция: LIME (Local Interpretable Model-agnostic Explanations).
1) Берём объект x_0. 2) Сэмплируем шумные точки x_i рядом. 3) Получаем f(x_i)
из чёрного ящика. 4) Взвешиваем точки по расстоянию до x_0 (близкие важнее).
5) Обучаем простую модель (например, Ridge) — её коэффициенты и есть объяснение.

Концепция: Веса близости.
Обычно гауссово ядро: w_i = exp(-||x_i - x_0||^2 / sigma^2). Чем ближе образец,
тем сильнее он влияет на локальную линейную модель.

Концепция: SHAP (SHapley Additive exPlanations).
Из теории игр: вклад фичи = средний прирост предсказания при добавлении этой
фичи во всевозможные подмножества других фич. Сумма SHAP-значений по всем
фичам = f(x) - E[f(X)]. Единственное объяснение, удовлетворяющее аксиомам.

Концепция: KernelSHAP.
Аппроксимация SHAP через взвешенную линейную регрессию по подмножествам фич.
Универсальная (как LIME), но веса другие — выводятся из аксиом Шепли.
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge

rng = np.random.default_rng(0)
X, y = make_classification(n_samples=400, n_features=5, n_informative=4, random_state=0)
clf = RandomForestClassifier(n_estimators=80, random_state=0).fit(X, y)

x0 = X[0]
p0 = clf.predict_proba(x0.reshape(1, -1))[0, 1]
print(f"Объясняем объект 0: P(y=1)={p0:.3f}")

# === LIME ===
N = 1000
sigma = 1.0
noise = rng.normal(scale=sigma, size=(N, X.shape[1]))
Xs = x0 + noise
fs = clf.predict_proba(Xs)[:, 1]
dist = np.linalg.norm(noise, axis=1)
w = np.exp(-(dist ** 2) / (2 * sigma ** 2))
surrogate = Ridge(alpha=1.0).fit(Xs, fs, sample_weight=w)
print(f"LIME coef (вклад каждой фичи): {surrogate.coef_.round(3)}")

# === KernelSHAP (упрощённая реализация) ===
# Для каждой фичи: сравниваем предсказания на парах (фича из x0) vs (фича из baseline).
baseline = X.mean(axis=0)
M = X.shape[1]
shap = np.zeros(M)
S = 200  # сэмплов масок
for _ in range(S):
    mask = rng.integers(0, 2, size=M).astype(bool)
    for j in range(M):
        # вклад фичи j: f с маской∪{j} минус f с маской\{j}
        m_in = mask.copy(); m_in[j] = True
        m_out = mask.copy(); m_out[j] = False
        x_in = np.where(m_in, x0, baseline)
        x_out = np.where(m_out, x0, baseline)
        shap[j] += clf.predict_proba(x_in.reshape(1, -1))[0, 1] - \
                   clf.predict_proba(x_out.reshape(1, -1))[0, 1]
shap /= S
print(f"SHAP вклад  (KernelSHAP-like): {shap.round(3)}")
print(f"Сумма SHAP ≈ f(x0) - E[f] = {shap.sum():.3f} vs {p0 - clf.predict_proba(baseline.reshape(1,-1))[0,1]:.3f}")
