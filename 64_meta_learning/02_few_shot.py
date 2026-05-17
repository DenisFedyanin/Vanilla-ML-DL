"""Раздел 64 — Мета-обучение.

Файл 2: Few-shot learning.

Концепция: N-way K-shot setup.
Эпизод задачи: выбираем N классов, в support set даём по K примеров на класс,
в query set — несколько примеров тех же N классов, которые нужно классифицировать.
Модель должна научиться 'учиться по нескольким примерам' (meta-learning).

Концепция: Prototypical Networks (Snell et al., 2017).
Эмбеддим support set, для каждого класса считаем ПРОТОТИП — среднее эмбеддинг
по K примерам. Query классифицируем по ближайшему прототипу в евклидовом смысле:
p(y=c|x) propto exp(-||f(x) - mu_c||^2).

Концепция: Matching Networks (Vinyals et al., 2016).
Идея: предсказание query = sum_i a(x, x_i) * y_i, где a — softmax по
cosine-сходству эмбеддингов. Это 'soft kNN' с обучаемым эмбеддером.

Концепция: MAML (Model-Agnostic Meta-Learning, Finn et al., 2017).
Обучаем такие начальные веса theta, что после 1-2 шагов SGD на support set новой
задачи модель хорошо работает на query set. Внешний цикл оптимизирует мета-параметры
по производной от внутреннего шага (second-order grad). FOMAML — без 2-го порядка.

Концепция: Reptile (Nichol et al., 2018).
Упрощение MAML: на каждой задаче делаем k шагов SGD из theta -> theta', затем
theta <- theta + eps * (theta' - theta). Не требует второго порядка.

Демо: на игрушечных 2D эмбеддингах строим прототипы трёх классов и классифицируем
query-точки. Для MAML/Reptile пишем псевдокод-комментарий.
"""
import numpy as np

rng = np.random.default_rng(0)

# === N-way K-shot: 3 класса по 5 примеров в support, 30 query ===
N, K = 3, 5
centers = np.array([[0.0, 0.0], [3.0, 0.0], [1.5, 2.6]])
support = np.concatenate([rng.normal(c, 0.5, size=(K, 2)) for c in centers])
support_y = np.repeat(np.arange(N), K)
query = np.concatenate([rng.normal(c, 0.5, size=(10, 2)) for c in centers])
query_y = np.repeat(np.arange(N), 10)

# === Prototypical Networks: прототип = среднее по support ===
prototypes = np.stack([support[support_y == c].mean(0) for c in range(N)])
# дистанции query <-> прототипы
d2 = ((query[:, None, :] - prototypes[None, :, :]) ** 2).sum(-1)
logits = -d2  # softmax от -d^2 = вероятности классов
pred = logits.argmax(1)
acc_proto = (pred == query_y).mean()
print(f"Prototypical Networks acc: {acc_proto:.3f}")
print("Прототипы классов:")
for c, mu in enumerate(prototypes):
    print(f"  class {c}: ({mu[0]:.2f}, {mu[1]:.2f}), теор центр {centers[c]}")

# === Matching Networks: soft kNN по cosine ===
def cos(a, b):
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return a @ b.T

sim = cos(query, support)  # (Nq, N*K)
# softmax по support, голосование по меткам
a = np.exp(sim - sim.max(1, keepdims=True))
a /= a.sum(1, keepdims=True)
one_hot = np.eye(N)[support_y]
probs = a @ one_hot
pred_m = probs.argmax(1)
acc_match = (pred_m == query_y).mean()
print(f"Matching Networks acc: {acc_match:.3f}")

# === MAML / Reptile (псевдокод-демо) ===
# Покажем Reptile-обновление на двух 'задачах' (предсказать центр прототипа линейной моделью).
def task_solve(W, X, y, lr=0.1, steps=10):
    """Внутренний цикл: несколько шагов SGD."""
    W = W.copy()
    for _ in range(steps):
        pred = X @ W
        W -= lr * X.T @ (pred - y) / len(X)
    return W

theta = rng.normal(size=(2, 2)) * 0.1
eps = 0.5
for outer in range(50):
    # сэмплируем задачу: классификация одного из 3 центров (one-hot регрессия)
    c = rng.integers(0, N)
    X_task = rng.normal(centers[c], 0.5, size=(8, 2))
    y_task = np.tile(np.eye(N)[c][:2], (8, 1))  # игрушечная цель
    theta_new = task_solve(theta, X_task, y_task)
    theta = theta + eps * (theta_new - theta)  # Reptile update
print(f"Reptile final ||theta||: {np.linalg.norm(theta):.3f}")
print("MAML: то же, но с производной по внешнему лоссу через внутренний шаг (2nd order).")
