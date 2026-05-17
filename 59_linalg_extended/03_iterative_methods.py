"""Раздел 59 — Линейная алгебра.

Файл 3: Итерационные методы.

Концепция: Power iteration.
Для матрицы A с уникальным по модулю максимальным собств. значением:
v_{k+1} = A v_k / ||A v_k|| сходится к собственному вектору этого значения.
Применяется в PageRank, PCA top-k.

Концепция: Krylov subspace.
K_m(A,b) = span{b, Ab, A^2 b, ..., A^{m-1} b}. Многие итерационные методы
строят аппроксимации в Krylov-подпространстве, не материализуя A целиком.

Концепция: Lanczos.
Для симметричной A строит ортонормированный базис Q_m в K_m(A,q1) такой что
Q_m^T A Q_m = T_m — трехдиагональная. Собств. значения T_m аппроксимируют собств. A.

Концепция: Arnoldi.
Обобщение Lanczos на несимметричные A. Дает Q_m^T A Q_m = H_m — верхняя Хессенберга.
Используется в GMRES, eigs (numpy/sparse).

Концепция: CG как Krylov-метод.
Сопряженные градиенты строят решение в K_k(A, r0) и минимизируют энергию ||x-x*||_A.
"""
import numpy as np

rng = np.random.default_rng(0)

# Симметричная матрица для power/Lanczos
A = np.diag([5.0, 3.0, 1.0, 0.5])
A = rng.normal(size=(4, 4))
A = A + A.T  # симметризуем

# === Power iteration ===
v = rng.normal(size=4); v /= np.linalg.norm(v)
for _ in range(50):
    v = A @ v
    v /= np.linalg.norm(v)
lam_top = v @ A @ v
true_lams = np.linalg.eigvalsh(A)
print(f"Power: lam_max≈{lam_top:.4f}, true |lam|max = {np.abs(true_lams).max():.4f}")

# === Krylov-базис {b, Ab, A^2 b, A^3 b} ===
b = rng.normal(size=4); b /= np.linalg.norm(b)
K = np.column_stack([b, A @ b, A @ A @ b, A @ A @ A @ b])
print(f"Krylov K_4: ранг = {np.linalg.matrix_rank(K)}")

# === Lanczos (3 шага) на симм. A ===
m = 3
Q = np.zeros((4, m + 1))
alpha = np.zeros(m)
beta = np.zeros(m + 1)
Q[:, 0] = b
for j in range(m):
    w = A @ Q[:, j] - beta[j] * (Q[:, j - 1] if j > 0 else 0)
    alpha[j] = Q[:, j] @ w
    w = w - alpha[j] * Q[:, j]
    # реортогонализация
    w -= Q[:, :j + 1] @ (Q[:, :j + 1].T @ w)
    beta[j + 1] = np.linalg.norm(w)
    if beta[j + 1] < 1e-12:
        break
    Q[:, j + 1] = w / beta[j + 1]

T = np.diag(alpha) + np.diag(beta[1:m], 1) + np.diag(beta[1:m], -1)
ritz = np.linalg.eigvalsh(T)
print(f"Lanczos m=3: Ritz vals = {ritz.round(3)}")
print(f"Сравнение с истинными eigvals: {true_lams.round(3)}")

# === Arnoldi-концепт: на несимметричной B ===
B = rng.normal(size=(4, 4))
q = b / np.linalg.norm(b)
Qa = [q]
H = np.zeros((4, 3))
for j in range(3):
    w = B @ Qa[j]
    for i in range(j + 1):
        H[i, j] = Qa[i] @ w
        w -= H[i, j] * Qa[i]
    H[j + 1, j] = np.linalg.norm(w)
    if H[j + 1, j] > 1e-12:
        Qa.append(w / H[j + 1, j])
print(f"Arnoldi H[:3,:3] eigvals = {np.linalg.eigvals(H[:3,:3]).round(3)}")

# === CG как Krylov: уже было в файле 04 — здесь только связь ===
print("CG строит x_k in K_k(A, r0) минимизируя ||x-x*||_A.")
