"""Раздел 67 — Online learning.

Файл 1: Online SGD, Perceptron, PA, FTRL.

Концепция: Online learning.
Данные приходят по одному (или маленьким батчем), модель обновляется на лету
и НЕ хранит весь датасет. Применения: стриминг кликов, рекламные ставки,
финтех. Метрика — кумулятивная ошибка / regret.

Концепция: Online Perceptron (Rosenblatt, 1958).
При ошибке обновляем w <- w + y * x (где y in {-1,+1}). При правильной классификации
не делаем ничего. Сходится за конечное число шагов, если данные линейно разделимы.

Концепция: Online SGD.
На каждом шаге: w <- w - eta * grad L(w; x_t, y_t). Самая базовая схема для
любого выпуклого лосса (log-loss, hinge, MSE). Часто с убывающим LR eta_t = 1/sqrt(t).

Концепция: Passive-Aggressive (PA, Crammer 2006).
Если предсказание уже правильно с маржой >= 1 — НИЧЕГО (passive). Иначе — делаем
минимальное изменение w, при котором лосс становится 0 (aggressive). Закрытое
решение: w <- w + tau * y * x, tau = max(0, 1 - y*<w,x>) / ||x||^2 для PA.
Варианты PA-I, PA-II ограничивают tau через параметр C.

Концепция: FTRL (Follow-The-Regularized-Leader, McMahan 2013).
w_{t+1} = argmin sum_{s<=t} L_s(w) + R(w). Накапливаем градиенты, на каждом шаге
решаем задачу с регуляризацией (часто L1+L2). FTRL-Proximal — стандарт в Google Ads,
даёт разреженные веса при L1.

Концепция: Hoeffding Trees (VFDT, концепт).
Решающее дерево для стрима: накапливаем статистики по каждому атрибуту, и как только
по неравенству Хёффдинга уверены, что лучший сплит действительно лучший — делаем сплит.
"""
import numpy as np

rng = np.random.default_rng(0)
# === поток: 2D точки, метки sign(x1 + x2) ===
N, d = 300, 2
X = rng.normal(size=(N, d))
y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)

# === Online Perceptron ===
w_p = np.zeros(d)
errs_p = 0
for t in range(N):
    if y[t] * (w_p @ X[t]) <= 0:
        w_p += y[t] * X[t]
        errs_p += 1
print(f"Perceptron: ошибок за поток = {errs_p}/{N}, w={np.round(w_p, 2)}")

# === Online SGD c log-loss ===
w_sgd = np.zeros(d)
errs_sgd = 0
for t in range(N):
    s = w_sgd @ X[t]
    pred = 1 if s > 0 else -1
    if pred != y[t]:
        errs_sgd += 1
    # grad log-loss для y in {-1,+1}: -y*x / (1 + exp(y*s))
    grad = -y[t] * X[t] / (1 + np.exp(y[t] * s))
    eta = 1.0 / np.sqrt(t + 1)
    w_sgd -= eta * grad
print(f"Online SGD : ошибок = {errs_sgd}/{N}, w={np.round(w_sgd, 2)}")

# === Passive-Aggressive (PA-I) ===
w_pa = np.zeros(d)
errs_pa = 0
C = 1.0
for t in range(N):
    s = w_pa @ X[t]
    if y[t] * s <= 0:
        errs_pa += 1
    loss = max(0, 1 - y[t] * s)
    tau = min(C, loss / (X[t] @ X[t] + 1e-12))
    w_pa += tau * y[t] * X[t]
print(f"Passive-Aggressive (PA-I): ошибок = {errs_pa}/{N}, w={np.round(w_pa, 2)}")

# === FTRL-Proximal (упрощённо: с L2) ===
# z_t = sum grad_s; w_{t+1} = -z_t / (lambda + 1/eta)
lam = 0.5
eta = 0.1
z = np.zeros(d)
errs_ftrl = 0
for t in range(N):
    w_ftrl = -z / (lam + 1 / eta)
    s = w_ftrl @ X[t]
    pred = 1 if s > 0 else -1
    if pred != y[t]:
        errs_ftrl += 1
    grad = -y[t] * X[t] / (1 + np.exp(y[t] * s))
    z += grad
print(f"FTRL (с L2): ошибок = {errs_ftrl}/{N}, w={np.round(w_ftrl, 2)}")

print("\nИтог: алгоритмы 'учатся на ходу', не хранят весь датасет.")
print("PA даёт минимально-агрессивный шаг; FTRL — разреженные веса при L1.")
