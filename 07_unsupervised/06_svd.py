"""Концепция 69: SVD - сингулярное разложение.

Любую матрицу можно представить как A = U * Sigma * V^T.
Используется в PCA, в рекомендательных системах, в сжатии изображений.
"""
import numpy as np

rng = np.random.default_rng(0)
A = rng.normal(size=(5, 3))

U, s, Vt = np.linalg.svd(A, full_matrices=False)
print("Сингулярные числа:", s)
A_back = U @ np.diag(s) @ Vt
print("Восстановили ли A? max abs diff =", np.max(np.abs(A - A_back)))

# Сжатие: оставим только 1 главное направление
A_rank1 = U[:, :1] @ np.diag(s[:1]) @ Vt[:1]
print("Ошибка приближения rank-1:", np.linalg.norm(A - A_rank1))
