"""Раздел 64 — Мета-обучение.

Файл 1: Transfer learning (перенос знаний).

Концепция: Transfer learning.
Идея: обучили модель на большой задаче A (например, ImageNet), а затем переиспользуем
её представления для задачи B, где данных мало. Это экономит данные и время.

Концепция: Feature extraction (заморозка энкодера).
Берём предобученный энкодер (нижние слои), ЗАМОРАЖИВАЕМ его веса, поверх ставим новую
'голову' (linear layer) и обучаем только её. Энкодер работает как фиксированный
извлекатель признаков.

Концепция: Fine-tuning.
После фазы feature extraction размораживаем часть/все слои энкодера и доучиваем
ВСЁ вместе с очень маленьким learning rate, чтобы не разрушить предобученные веса.

Концепция: Domain adaptation.
Обучили на source-домене, нужно работать на target-домене с другой статистикой.
Подходы: подгонка статистик батча, adversarial обучение признаков, доменный дискриминатор.
Здесь демонстрируем простейшее: дообучение головы под новый домен.

Демо: 'предобучаем' линейный энкодер W на задаче A, потом для задачи B
обучаем только новую голову u (feature extraction), а затем размораживаем W
с маленьким LR (fine-tuning) и сравниваем качество.
"""
import numpy as np

rng = np.random.default_rng(0)

# === предобучение на задаче A: y_A = sign(W_true @ x) ===
d, h = 8, 4
W_true_A = rng.normal(size=(h, d))
X_A = rng.normal(size=(2000, d))
y_A = (X_A @ W_true_A[0] > 0).astype(float) * 2 - 1
# простая линейная регрессия с h-мерным узким горлышком (h<d)
# учим энкодер W (d->h) и голову v (h) как одну линейную модель v @ W
W = rng.normal(size=(h, d)) * 0.1
v_A = rng.normal(size=h) * 0.1
for _ in range(200):
    z = X_A @ W.T  # (n,h)
    pred = z @ v_A
    err = pred - y_A
    grad_v = z.T @ err / len(X_A)
    grad_W = np.outer(v_A, X_A.T @ err / len(X_A))
    v_A -= 0.05 * grad_v
    W -= 0.05 * grad_W
acc_A = ((X_A @ W.T @ v_A) * y_A > 0).mean()
print(f"Pretrain task A acc: {acc_A:.3f}")

# === задача B (родственная, мало данных): y_B = sign(W_true @ x проецированное иначе) ===
v_true_B = rng.normal(size=h)
X_B = rng.normal(size=(60, d))
y_B = ((X_B @ W_true_A.T) @ v_true_B > 0).astype(float) * 2 - 1
X_test = rng.normal(size=(500, d))
y_test = ((X_test @ W_true_A.T) @ v_true_B > 0).astype(float) * 2 - 1

# --- feature extraction: W заморожен, учим только голову u ---
Z_B = X_B @ W.T
u = np.zeros(h)
for _ in range(300):
    err = Z_B @ u - y_B
    u -= 0.1 * Z_B.T @ err / len(X_B)
acc_fe = ((X_test @ W.T @ u) * y_test > 0).mean()
print(f"Feature extraction acc: {acc_fe:.3f}")

# --- fine-tuning: размораживаем W с очень малым LR, голову с обычным ---
W_ft = W.copy()
u_ft = u.copy()
for _ in range(300):
    z = X_B @ W_ft.T
    err = z @ u_ft - y_B
    u_ft -= 0.1 * z.T @ err / len(X_B)
    W_ft -= 0.001 * np.outer(u_ft, X_B.T @ err / len(X_B))  # маленький LR
acc_ft = ((X_test @ W_ft.T @ u_ft) * y_test > 0).mean()
print(f"Fine-tuning acc: {acc_ft:.3f}")

# --- baseline: обучить с нуля при малых данных ---
W_s = rng.normal(size=(h, d)) * 0.1
u_s = rng.normal(size=h) * 0.1
for _ in range(300):
    z = X_B @ W_s.T
    err = z @ u_s - y_B
    u_s -= 0.05 * z.T @ err / len(X_B)
    W_s -= 0.05 * np.outer(u_s, X_B.T @ err / len(X_B))
acc_scratch = ((X_test @ W_s.T @ u_s) * y_test > 0).mean()
print(f"From-scratch acc: {acc_scratch:.3f}")

# === Domain adaptation (концепт) ===
# Сдвинем X_B на константу (имитация смены домена) и просто пересчитаем mean.
X_B_shift = X_B + 0.5
mean_src = X_A.mean(0)
mean_tgt = X_B_shift.mean(0)
X_B_aligned = X_B_shift - (mean_tgt - mean_src)  # простейшее выравнивание
print(f"Domain shift before: {np.linalg.norm(mean_tgt - mean_src):.3f}, "
      f"after align: {np.linalg.norm(X_B_aligned.mean(0) - mean_src):.3f}")
