"""Раздел 59 — Линейная алгебра (расширение).

Файл 1: Разложения матриц.

Концепция: LU-разложение.
A = L U, L нижнетреугольная с единичной диагональю, U верхнетреугольная.
Используется для решения линейных систем за O(n^2) после O(n^3) на разложение.

Концепция: QR-разложение.
A = Q R, Q ортогональна, R верхнетреугольная. Применяется в МНК (численно устойчиво)
и в QR-алгоритме поиска собственных значений.

Концепция: Cholesky.
Для симметричной положительно определенной (SPD) A: A = L L^T, L нижнетреугольная.
Вдвое быстрее LU, используется в ковариационных задачах и BLR.

Концепция: Eigendecomposition.
Для квадратной A: A = V Lambda V^{-1} (если есть n линейно независимых собств. векторов).
Для симметричной: A = Q Lambda Q^T, Q ортогональна.

Концепция: Schur-разложение.
A = Q T Q^*, Q унитарна, T верхнетреугольная. Существует для любой комплексной A.
Используется внутри QR-алгоритма как стабильное представление.
"""
import numpy as np

rng = np.random.default_rng(0)

# === Ручное LU (без перестановок) ===
A = np.array([[2.0, 1.0, 1.0],
              [4.0, 3.0, 3.0],
              [8.0, 7.0, 9.0]])
n = A.shape[0]
L = np.eye(n); U = A.copy()
for k in range(n - 1):
    for i in range(k + 1, n):
        L[i, k] = U[i, k] / U[k, k]
        U[i, k:] -= L[i, k] * U[k, k:]
print(f"LU: ||A - LU|| = {np.linalg.norm(A - L @ U):.2e}")

# === QR через numpy ===
Q, R = np.linalg.qr(A)
print(f"QR: ||Q^T Q - I||={np.linalg.norm(Q.T @ Q - np.eye(n)):.2e}, ||A-QR||={np.linalg.norm(A - Q@R):.2e}")

# === Ручной Cholesky на SPD ===
M = np.array([[4.0, 2.0, 0.0],
              [2.0, 5.0, 1.0],
              [0.0, 1.0, 3.0]])
n = M.shape[0]
Lc = np.zeros_like(M)
for i in range(n):
    for j in range(i + 1):
        s = M[i, j] - Lc[i, :j] @ Lc[j, :j]
        Lc[i, j] = np.sqrt(s) if i == j else s / Lc[j, j]
print(f"Cholesky: ||M - L L^T|| = {np.linalg.norm(M - Lc @ Lc.T):.2e}")

# === Eigendecomposition (симметричная) ===
w, V = np.linalg.eigh(M)
recon = V @ np.diag(w) @ V.T
print(f"Eigen sym: eigvals={w.round(3)}, ||M - V D V^T||={np.linalg.norm(M - recon):.2e}")

# === Schur (концепт): в numpy нет напрямую, но T = Q^T A Q верхнетреугольная для нормальной A
# Покажем на симметричной A: T=diag(w), Q=V собственных векторов.
T_check = V.T @ M @ V
print(f"Schur sym: верхнетреуг. часть = diag(w): {np.allclose(T_check, np.diag(w))}")
