"""Раздел 22 — Расширенное снижение размерности. Файл 1: LDA и QDA.

Концепция 640: Linear Discriminant Analysis (LDA).
Supervised метод. Ищет проекцию, максимизирующую отношение межклассовой
дисперсии к внутриклассовой: J(w) = w^T S_B w / w^T S_W w. Решение — обобщённая
задача на собственные значения S_W^{-1} S_B.

Концепция 641: Within-class scatter S_W.
S_W = sum_c sum_{x in c} (x - mu_c)(x - mu_c)^T.
«Насколько разбросаны точки внутри своих классов».

Концепция 642: Between-class scatter S_B.
S_B = sum_c N_c * (mu_c - mu)(mu_c - mu)^T.
«Насколько разнесены центры классов от общего среднего».

Концепция 643: Размерность LDA-проекции.
Для C классов LDA даёт максимум C-1 компонент (ранг S_B = C-1).

Концепция 644: Quadratic Discriminant Analysis (QDA).
Допускает разные ковариационные матрицы для классов: граница не линейная,
а квадратичная. Гибче, но требует больше данных (оценка C матриц).
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

X, y = load_iris(return_X_y=True)
classes = np.unique(y)
D = X.shape[1]
mu = X.mean(axis=0)

# === 641-642: ручные scatter-матрицы ===
S_W = np.zeros((D, D))
S_B = np.zeros((D, D))
for c in classes:
    Xc = X[y == c]
    muc = Xc.mean(axis=0)
    S_W += (Xc - muc).T @ (Xc - muc)
    diff = (muc - mu).reshape(-1, 1)
    S_B += len(Xc) * (diff @ diff.T)

print(f"641 S_W diag: {np.diag(S_W).round(2).tolist()}")
print(f"642 S_B diag: {np.diag(S_B).round(2).tolist()}")

# === 640: решаем обобщённую задачу через eigh ===
A = np.linalg.solve(S_W, S_B)
vals, vecs = np.linalg.eig(A)
order = np.argsort(-vals.real)
W = vecs[:, order[:2]].real  # 2 компоненты
X_lda = X @ W
print(f"640 LDA-проекция в 2D: shape={X_lda.shape}, top eigvals="
      f"{vals.real[order[:3]].round(2).tolist()}")

# === 643: число компонент = C-1 ===
print(f"643 классов={len(classes)}, max LDA-компонент={len(classes) - 1}")

# === 644: QDA через sklearn ===
qda = QuadraticDiscriminantAnalysis().fit(X, y)
print(f"644 QDA: train accuracy={qda.score(X, y):.3f}")
