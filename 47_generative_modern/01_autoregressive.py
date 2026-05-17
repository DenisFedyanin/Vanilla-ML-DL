"""Раздел 47 — Современные генеративные модели.

Файл 1: автогрессивные модели.

Концепция: AR-модели.
p(x) = П_t p(x_t | x_<t). Точная likelihood, легко считать log p(x).
Минус: семплинг последователен (медленно).

Концепция: PixelRNN.
LSTM по пикселям изображения (row-by-row). Условное распределение каждого
пикселя зависит от всех ранее сгенерированных.

Концепция: PixelCNN.
Заменяет RNN на masked conv: маска зануляет "будущие" пиксели. Параллельное
обучение, но семплинг всё равно по одному пикселю.

Концепция: WaveNet.
Аудио-генерация. Стек dilated causal convs: receptive field экспоненциально
растёт с глубиной, при этом O(N) параметров на слой. Causal = не смотрит в будущее.

Концепция: Causal masking.
В свёртке/attention обеспечиваем, что выход на позиции t зависит только
от позиций <=t. Для conv: сдвиг ядра вправо. Для attention: маска треугольника.
"""
import numpy as np

rng = np.random.default_rng(0)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


# Игрушечные данные: последовательности длины T бернулли с автокорреляцией
T = 8
N = 1000
data = np.zeros((N, T), dtype=int)
data[:, 0] = rng.uniform(size=N) < 0.5
for t in range(1, T):
    p = sigmoid(0.5 * (2 * data[:, t - 1] - 1) + 0.3 * (2 * data[:, max(0, t - 2)] - 1))
    data[:, t] = rng.uniform(size=N) < p

# Tiny AR-модель: p(x_t=1 | x_{t-1}, x_{t-2}) = sigmoid(a + b*x_{t-1} + c*x_{t-2})
a, b, c = 0.0, 0.0, 0.0
lr = 0.05
for epoch in range(50):
    grad_a = grad_b = grad_c = 0.0
    for t in range(2, T):
        z = a + b * data[:, t - 1] + c * data[:, t - 2]
        p = sigmoid(z)
        diff = data[:, t] - p
        grad_a += diff.mean()
        grad_b += (diff * data[:, t - 1]).mean()
        grad_c += (diff * data[:, t - 2]).mean()
    a += lr * grad_a
    b += lr * grad_b
    c += lr * grad_c

print(f"Выученные параметры: a={a:.2f}, b={b:.2f}, c={c:.2f}")
print(f"(истинные: a=0, b=1.0 (sum 0.5+...), c=0.6)")

# Sampling — последовательно
samples = np.zeros((10, T), dtype=int)
samples[:, 0] = rng.uniform(size=10) < 0.5
for t in range(1, T):
    prev = samples[:, t - 1]
    prev2 = samples[:, max(0, t - 2)] if t >= 2 else np.zeros(10, dtype=int)
    p = sigmoid(a + b * prev + c * prev2)
    samples[:, t] = rng.uniform(size=10) < p
print("Сгенерированные последовательности (10 шт.):")
print(samples)

# Causal mask demo: причинная маска для attention/conv
mask = np.tril(np.ones((6, 6), dtype=int))
print("Causal mask 6x6 (1 = можно смотреть):")
print(mask)

# Dilated causal conv WaveNet: receptive field 2^L
for L in [1, 2, 3, 4]:
    rf = 1 + sum(2 ** l for l in range(L))
    print(f"WaveNet {L} dilated layers (1,2,4,...): RF = {rf}")
