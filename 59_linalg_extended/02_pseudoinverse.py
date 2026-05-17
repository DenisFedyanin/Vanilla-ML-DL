"""Раздел 59 — Линейная алгебра.

Файл 2: Псевдообратная матрица и связанные понятия.

Концепция: Moore-Penrose pseudoinverse.
Для A = U S V^T (SVD): A^+ = V S^+ U^T, где S^+ заменяет ненулевые sigma_i на 1/sigma_i.
Существует и единственна для любой A. Удовлетворяет 4 свойствам Moore-Penrose.

Концепция: МНК через псевдообратную.
Решение argmin ||Ax-b|| даётся x = A^+ b. Если A полного столбцового ранга:
A^+ = (A^T A)^{-1} A^T (нормальные уравнения).

Концепция: Число обусловленности.
cond(A) = sigma_max / sigma_min. Чем больше, тем более чувствительно решение Ax=b к ошибкам.
Для cond=10^k можно потерять около k значащих цифр.

Концепция: Rank-revealing разложения.
QR с поворотом столбцов или SVD позволяют выявить численный ранг (число sigma_i выше порога).
Использется в low-rank аппроксимации и регуляризации.
"""
import numpy as np

rng = np.random.default_rng(0)

# Прямоугольная A (тонкая, переопределенная)
n, d = 20, 5
A = rng.normal(size=(n, d))

# === SVD-разложение и pseudoinverse вручную ===
U, S, Vt = np.linalg.svd(A, full_matrices=False)
S_pinv = np.diag([1 / s if s > 1e-12 else 0 for s in S])
A_pinv = Vt.T @ S_pinv @ U.T

print(f"||A^+ - np.pinv(A)|| = {np.linalg.norm(A_pinv - np.linalg.pinv(A)):.2e}")
# Свойство: A A^+ A = A
print(f"A A^+ A == A: {np.allclose(A @ A_pinv @ A, A)}")

# === МНК через pinv ===
b = rng.normal(size=n)
x_pinv = A_pinv @ b
# нормальные уравнения дают то же самое
x_normal = np.linalg.solve(A.T @ A, A.T @ b)
print(f"||x_pinv - x_normal|| = {np.linalg.norm(x_pinv - x_normal):.2e}")

# === Число обусловленности ===
cond = S[0] / S[-1]
print(f"cond(A) = {cond:.2f}, np.linalg.cond = {np.linalg.cond(A):.2f}")

# Плохо обусловленная: матрица Гильберта 6x6
H = 1.0 / (np.arange(1, 7)[:, None] + np.arange(1, 7)[None, :] - 1)
print(f"cond(Hilbert6) ≈ {np.linalg.cond(H):.2e} -> решение очень неустойчиво")

# === Rank-revealing: матрица ранга 3, но размер 6x6 ===
B = rng.normal(size=(6, 3)) @ rng.normal(size=(3, 6))
s = np.linalg.svd(B, compute_uv=False)
print(f"Сингулярные значения B: {s.round(4)}")
print(f"Численный ранг (порог 1e-10) = {np.sum(s > 1e-10)}")
