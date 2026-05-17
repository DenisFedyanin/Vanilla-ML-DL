"""Концепция 60: Перцептрон Розенблатта (1957).

Прародитель всех нейросетей. Работает только для линейно разделимых задач.
Правило: при ошибке w := w + y * x.
"""
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(60, 2))
y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)

w = np.zeros(2); b = 0.0
for epoch in range(30):
    errors = 0
    for i in range(len(X)):
        if y[i] * (X[i] @ w + b) <= 0:
            w += y[i] * X[i]; b += y[i]
            errors += 1
    if errors == 0:
        print(f"Сошёлся за {epoch + 1} эпох"); break

acc = (np.sign(X @ w + b) == y).mean()
print(f"Accuracy: {acc:.3f}")
