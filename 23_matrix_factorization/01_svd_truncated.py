"""Раздел 23 — Матричные разложения. Файл 1: SVD и Truncated SVD.

Концепция 655: SVD — Singular Value Decomposition.
Любая матрица A (m x n) представима как A = U S V^T, где U, V ортогональные,
S диагональна с неотрицательными сингулярными числами. Самое универсальное
разложение в линейной алгебре.

Концепция 656: Truncated SVD (rank-k approximation).
Берём первые k сингулярных чисел и соответствующие столбцы U, V:
A_k = U_k S_k V_k^T. По теореме Эккарта-Янга это лучшее приближение ранга k
по фробениус- и спектральной норме.

Концепция 657: Truncated SVD как LSA / LSI.
В NLP применяется к терм-документной матрице — даёт «латентные семантические»
направления (LSA). В sklearn это `sklearn.decomposition.TruncatedSVD`.

Концепция 658: Сжатие данных через rank-k.
Параметров в A_k: m*k + k + k*n ≈ k*(m+n) против m*n у полной матрицы.
Чем меньше k — тем сильнее сжатие и тем больше потеря.
"""
import numpy as np
from sklearn.decomposition import TruncatedSVD

rng = np.random.default_rng(0)

# === 655: полное SVD ===
A = rng.normal(size=(30, 30))
U, S, Vt = np.linalg.svd(A, full_matrices=False)
print(f"655 SVD 30x30: top-5 sing values={S[:5].round(2).tolist()}, "
      f"reconstruction err={np.linalg.norm(A - U @ np.diag(S) @ Vt):.2e}")

# === 656-658: rank-k приближение и ошибка ===
norm_A = np.linalg.norm(A)
for k in (1, 5, 10, 20, 30):
    Ak = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    err = np.linalg.norm(A - Ak) / norm_A
    params = k * (A.shape[0] + A.shape[1])
    full_params = A.shape[0] * A.shape[1]
    print(f"656-658 rank-{k:>2}: relative err={err:.3f}, "
          f"параметров {params}/{full_params}")

# === 657: TruncatedSVD через sklearn (LSA) ===
ts = TruncatedSVD(n_components=5, random_state=0).fit_transform(A)
print(f"657 TruncatedSVD sklearn: shape={ts.shape}, "
      f"explained_variance_ratio суммарно={TruncatedSVD(5,random_state=0).fit(A).explained_variance_ratio_.sum():.2f}")
