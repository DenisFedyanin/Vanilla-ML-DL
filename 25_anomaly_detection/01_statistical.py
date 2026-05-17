"""Раздел 25 — Поиск аномалий. Файл 1: статистические методы.

Концепция 679: Z-score.
z_i = (x_i - mean)/std. Аномалия — |z| > 3 (правило трёх сигм). Прост, но
чувствителен к самим выбросам (они портят mean и std).

Концепция 680: IQR-метод (boxplot rule).
Q1, Q3 — 25-й и 75-й перцентили; IQR = Q3 - Q1. Выбросы — вне
[Q1 - 1.5*IQR, Q3 + 1.5*IQR]. Робастен, не требует нормальности.

Концепция 681: MAD — Median Absolute Deviation.
MAD = median(|x - median(x)|). Робастная замена std; модифицированный z-score
z' = 0.6745 * (x - median) / MAD; порог |z'| > 3.5.

Концепция 682: Modified Z-score.
То же, что в 681. Названо отдельно (Iglewicz & Hoaglin), широко применяется
в QC и финансах. Не «ломается» от единичных выбросов.

Концепция 683: Mahalanobis distance.
Для многомерных данных учитывает ковариацию признаков:
d_M(x) = sqrt((x-mu)^T Sigma^{-1} (x-mu)).
Аномалия — большая d_M. Эквивалент масштабирования к единичной ковариации.
"""
import numpy as np

rng = np.random.default_rng(0)

# === 679: Z-score на 1D ===
x = np.concatenate([rng.normal(0, 1, 100), np.array([8.0, -7.0, 9.5])])
z = (x - x.mean()) / x.std()
out_z = np.where(np.abs(z) > 3)[0]
print(f"679 Z-score: найдено выбросов={len(out_z)}, индексы={out_z.tolist()}")

# === 680: IQR ===
Q1, Q3 = np.percentile(x, [25, 75])
iqr = Q3 - Q1
out_iqr = np.where((x < Q1 - 1.5 * iqr) | (x > Q3 + 1.5 * iqr))[0]
print(f"680 IQR: Q1={Q1:.2f}, Q3={Q3:.2f}, выбросов={len(out_iqr)}")

# === 681: MAD ===
med = np.median(x)
mad = np.median(np.abs(x - med))
out_mad = np.where(np.abs(x - med) > 3 * 1.4826 * mad)[0]
print(f"681 MAD: median={med:.2f}, MAD={mad:.2f}, выбросов={len(out_mad)}")

# === 682: Modified Z-score ===
mz = 0.6745 * (x - med) / mad
out_mz = np.where(np.abs(mz) > 3.5)[0]
print(f"682 Modified Z: выбросов={len(out_mz)}, индексы={out_mz.tolist()}")

# === 683: Mahalanobis distance — 2D данные с одним выбросом ===
X = np.column_stack([rng.normal(0, 1, 100), rng.normal(0, 1, 100)])
X = np.vstack([X, [[6, 6], [-5, 5]]])  # 2 выброса
mu = X.mean(axis=0)
S = np.cov(X.T)
S_inv = np.linalg.inv(S)
diff = X - mu
d_M = np.sqrt(np.einsum("ni,ij,nj->n", diff, S_inv, diff))
out_M = np.where(d_M > np.percentile(d_M, 95))[0]
print(f"683 Mahalanobis: top-3 max d_M={np.sort(d_M)[-3:].round(2).tolist()}, "
      f">95% перцентиль: {len(out_M)} точек")
