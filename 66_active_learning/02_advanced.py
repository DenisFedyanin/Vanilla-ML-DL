"""Раздел 66 — Active Learning.

Файл 2: Продвинутые стратегии.

Концепция: Query-by-committee (QBC).
Обучаем K моделей-членов комитета (бутстрап / разные seed). Запрашиваем
объект, на котором они БОЛЬШЕ ВСЕГО НЕ СОГЛАСНЫ (vote entropy или KL-дивергенция
между предсказаниями). Идея: где модели расходятся — там информативно.

Концепция: Expected Model Change.
Выбираем точку, добавление которой максимально изменит параметры модели
(норма градиента ||grad L(x, y_pseudo)||). На практике дорогостоящее.

Концепция: Expected Error Reduction (EER).
Выбираем x, после добавления которого ожидаемая ошибка на пуле минимизируется:
x* = argmin_x sum_y P(y|x) * Error(theta + (x,y)). Очень дорого: для каждого
кандидата нужно дообучить модель и оценить пул.

Концепция: Density-Weighted methods.
Минус uncertainty sampling: выбирает выбросы. Решение — взвешивать score
на плотность: phi(x) = unc(x) * (среднее сходство с пулом)^beta.
Выбираем 'неуверенные, но репрезентативные' точки.

Концепция: BatchBALD (Bayesian Active Learning by Disagreement, batch версия).
Для глубоких сетей с MC-dropout: выбираем БАТЧ точек, максимизирующих взаимную
информацию I(y_batch; theta). В отличие от обычного BALD, учитывает корреляции
внутри батча (не берёт дубликаты).

Демо: QBC на маленьком пуле с 5 моделями (разные seeds на бутстрапе).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

rng = np.random.default_rng(0)
X, y = make_classification(n_samples=300, n_features=4, n_informative=3,
                           n_redundant=0, n_classes=3, n_clusters_per_class=1,
                           random_state=0)

idx = rng.permutation(len(X))
labeled = list(idx[:15])
pool = list(idx[15:])

# === обучение комитета ===
def train_committee(K=5):
    models = []
    n = len(labeled)
    for k in range(K):
        local = np.random.default_rng(k)
        sub = local.choice(labeled, size=n, replace=True)
        # гарантируем минимум 2 класса в подмножестве
        if len(np.unique(y[sub])) < 2:
            sub = labeled
        m = LogisticRegression(max_iter=300).fit(X[sub], y[sub])
        models.append(m)
    return models

models = train_committee()

# === Vote entropy ===
preds = np.stack([m.predict(X[pool]) for m in models])  # (K, |pool|)
K, n_pool = preds.shape
n_classes = 3
votes = np.zeros((n_pool, n_classes))
for k in range(K):
    for c in range(n_classes):
        votes[:, c] += (preds[k] == c)
votes /= K
eps = 1e-12
vote_entropy = -(votes * np.log(votes + eps)).sum(1)
best_qbc = pool[int(np.argmax(vote_entropy))]
print(f"QBC: vote entropy top5 = {np.round(np.sort(vote_entropy)[-5:], 3)}")
print(f"Выбран объект #{best_qbc}, истинный класс = {y[best_qbc]}")

# === Density-weighted: unc * среднее косинусное сходство с пулом ===
probs = models[0].predict_proba(X[pool])
uncertainty = 1 - probs.max(1)
Xp = X[pool]
Xp_n = Xp / (np.linalg.norm(Xp, axis=1, keepdims=True) + 1e-12)
sim_matrix = Xp_n @ Xp_n.T
density = sim_matrix.mean(1)
score_dw = uncertainty * (density ** 1.0)
best_dw = pool[int(np.argmax(score_dw))]
print(f"Density-weighted: top score = {score_dw.max():.3f}, выбран #{best_dw}")

# === Expected Model Change (приближённо: норма градиента log-loss по w) ===
m = models[0]
probs_pool = m.predict_proba(X[pool])
# у LR градиент по w_c для (x, y) = (P(c|x) - I[y=c]) * x; берём усреднённую по y норму
norm_grad = np.zeros(n_pool)
for i, xi in enumerate(Xp):
    g = 0
    for c in range(n_classes):
        residual = probs_pool[i].copy()
        residual[c] -= 1
        g += probs_pool[i, c] * np.linalg.norm(np.outer(residual, xi))
    norm_grad[i] = g
best_emc = pool[int(np.argmax(norm_grad))]
print(f"Expected Model Change: max ||grad||={norm_grad.max():.2f}, выбран #{best_emc}")

# === Концепты EER и BatchBALD ===
print("\nКонцепты:")
print("  EER:       перебираем (x, y), считаем new ожидаемую ошибку на пуле — дорого.")
print("  BatchBALD: I(y_batch; theta) с MC-dropout, выбираем неоднообразный батч.")
