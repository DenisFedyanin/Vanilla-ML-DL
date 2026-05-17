"""Раздел 22 — Расширенное снижение размерности. Файл 2: факторные модели.

Концепция 645: Factor Analysis.
x = W*z + mu + eps, где z ~ N(0,I), eps ~ N(0,Psi) — независимый шум по
координатам. Похоже на PCA, но допускает разную дисперсию шума по признакам
(Psi диагональна, не сферична).

Концепция 646: Independent Component Analysis (ICA).
Раскладывает сигнал на статистически независимые источники, не просто
некоррелированные. Решает задачу blind source separation («коктейль-party»).
FastICA — самый популярный алгоритм.

Концепция 647: Non-negative Matrix Factorization (NMF).
X ≈ W * H, где X, W, H >= 0. Полезно, когда отрицательные значения не имеют
смысла (счётчики, частоты, спектры). Часто даёт «части целого» (parts-based).
"""
import numpy as np
from sklearn.decomposition import FactorAnalysis, FastICA, NMF
from sklearn.datasets import load_digits

rng = np.random.default_rng(0)

# === 645: Factor Analysis на digits ===
X, _ = load_digits(return_X_y=True)
fa = FactorAnalysis(n_components=5, random_state=0).fit(X)
print(f"645 FactorAnalysis: explained noise_variance[:5]="
      f"{fa.noise_variance_[:5].round(2).tolist()}, "
      f"components shape={fa.components_.shape}")

# === 646: ICA — разделение трёх «звуковых» сигналов ===
t = np.linspace(0, 5, 500)
s1 = np.sin(2 * t)             # синус
s2 = np.sign(np.sin(3 * t))    # квадратная волна
s3 = rng.uniform(-1, 1, size=t.shape)  # шум
S = np.column_stack([s1, s2, s3])
A = np.array([[1.0, 1.0, 1.0], [0.5, 2.0, 1.0], [1.5, 1.0, 2.0]])  # микширование
Xmix = S @ A.T
ica = FastICA(n_components=3, random_state=0).fit(Xmix)
S_est = ica.transform(Xmix)
# корреляция восстановленных компонент с исходными
corrs = []
for i in range(3):
    best = max(abs(np.corrcoef(S_est[:, i], S[:, j])[0, 1]) for j in range(3))
    corrs.append(best)
print(f"646 ICA восстановила 3 источника, лучшие корреляции="
      f"{[round(c, 2) for c in corrs]}")

# === 647: NMF (с неотрицательными данными) ===
X_pos = np.abs(X)  # digits и так >=0
nmf = NMF(n_components=5, init="nndsvd", max_iter=200, random_state=0).fit(X_pos)
W = nmf.transform(X_pos)
H = nmf.components_
err = np.linalg.norm(X_pos - W @ H) / np.linalg.norm(X_pos)
print(f"647 NMF на digits: W shape={W.shape}, H shape={H.shape}, "
      f"relative error={err:.3f}")
