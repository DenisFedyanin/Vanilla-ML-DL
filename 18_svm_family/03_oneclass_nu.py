"""Раздел 18 — Семейство SVM.

Файл 3: Nu-SVM и OneClassSVM (детекция аномалий).

Концепция 271: NuSVC.
Параметризация через nu в (0,1] вместо C. nu — верхняя граница доли
margin-нарушителей и нижняя граница доли support vectors. Более интерпретируемо.

Концепция 272: NuSVR.
То же для регрессии. nu контролирует долю объектов вне epsilon-трубки —
параметр epsilon при этом подбирается автоматически.

Концепция 273: OneClassSVM.
Обучается только на "нормальных" объектах, отделяет их от начала координат
в признаковом пространстве. Predict: +1 — нормальный, -1 — аномалия.
"""
import numpy as np
from sklearn.datasets import make_moons
from sklearn.svm import NuSVC, NuSVR, OneClassSVM

rng = np.random.default_rng(0)

# 271: NuSVC
X, y = make_moons(n_samples=200, noise=0.25, random_state=0)
nu_svc = NuSVC(nu=0.1, gamma="scale").fit(X, y)
print(f"271 NuSVC(nu=0.1) train acc={nu_svc.score(X, y):.3f}, n_support={int(nu_svc.n_support_.sum())}")

# 272: NuSVR
X_r = np.linspace(0, 10, 150).reshape(-1, 1)
y_r = np.sin(X_r.ravel()) + rng.normal(0, 0.2, 150)
nu_svr = NuSVR(nu=0.2, C=1.0).fit(X_r, y_r)
print(f"272 NuSVR(nu=0.2) R^2={nu_svr.score(X_r, y_r):.3f}, n_support={int(nu_svr.support_.size)}")

# 273: OneClassSVM
normal = rng.normal(size=(200, 2))
ocs = OneClassSVM(nu=0.1, gamma="scale").fit(normal)
test = np.vstack([rng.normal(size=(100, 2)), rng.uniform(-5, 5, size=(20, 2))])
preds = ocs.predict(test)
print(f"273 OneClassSVM: nominal {int((preds[:100] == 1).sum())}/100, аномалий поймано {int((preds[100:] == -1).sum())}/20")
