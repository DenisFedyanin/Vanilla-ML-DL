"""Раздел 18 — Семейство SVM.

Файл 2: разные ядра SVM.

Концепция 267: Linear kernel.
k(x, z) = x^T z. Эквивалентен обычной линейной модели; ничего "не подымает"
в признаковом пространстве. Быстро на больших n.

Концепция 268: Polynomial kernel.
k(x, z) = (gamma * x^T z + coef0)^degree. Учитывает взаимодействия признаков
до степени degree. Может переобучаться при больших степенях.

Концепция 269: RBF (Gaussian) kernel.
k(x, z) = exp(-gamma * ||x-z||^2). Локальный, "колоколообразный" — каждая
опорная точка действует как маленький детектор. По умолчанию в SVC.

Концепция 270: Sigmoid kernel.
k(x, z) = tanh(gamma * x^T z + coef0). Похож на однослойную нейронную сеть.
Не положительно полуопределён при всех параметрах, поэтому используется редко.
"""
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = make_moons(n_samples=300, noise=0.25, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

for kernel in ("linear", "poly", "rbf", "sigmoid"):
    svc = SVC(kernel=kernel, degree=3, gamma="scale", random_state=0).fit(X_tr, y_tr)
    acc = svc.score(X_te, y_te)
    tag = {"linear": 267, "poly": 268, "rbf": 269, "sigmoid": 270}[kernel]
    print(f"{tag} kernel={kernel:>8}: test acc={acc:.3f}, n_support={int(svc.n_support_.sum())}")
