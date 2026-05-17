"""Раздел 62 — Трюки обучения.

Файл 1: Точность и память.

Концепция: Mixed precision (FP16/BF16).
Хранить веса и градиенты в 16-битах -> 2x экономия памяти и ~2x ускорение на GPU с tensor cores.
FP16: 5 бит exp, 10 mantissa — узкий диапазон, нужен loss scaling. BF16: 8 exp, 7 mantissa —
тот же диапазон что FP32, обычно "бесшовно". Master weights в FP32.

Концепция: Gradient checkpointing.
Вместо хранения всех активаций forward — храним только часть; недостающие пересчитываем
во время backward. Память ~ O(sqrt(L)) вместо O(L), цена — ~33% доп. flops.

Концепция: Gradient accumulation.
Эффективный батч = большой, физический батч = маленький. Накапливаем grad по K мини-батчам,
делаем step один раз. Экономит память при сохранении статистики большого батча.
"""
import numpy as np

rng = np.random.default_rng(0)

# === Mixed precision: имитация FP16 округлением ===
x_fp32 = rng.normal(size=5).astype(np.float32)
x_fp16 = x_fp32.astype(np.float16)
print(f"FP32 веса: {x_fp32}")
print(f"FP16 веса: {x_fp16.astype(np.float32)}  (потеря {np.abs(x_fp32 - x_fp16.astype(np.float32)).max():.2e})")

# Loss scaling — умножаем loss перед backward на S, делим grad после
S = 1024
g_fp32 = rng.normal(size=5) * 1e-6  # очень маленькие grads
g_scaled = (g_fp32 * S).astype(np.float16)  # теперь FP16-репрезентабельны
g_back = g_scaled.astype(np.float32) / S
print(f"Loss scale (S={S}): сохранение grad точности OK -> ||diff||={np.linalg.norm(g_back - g_fp32):.2e}")

# === Gradient checkpointing: 3-слойный MLP, пересчет средних активаций ===
def layer(x, W): return np.tanh(x @ W)

W1, W2, W3 = rng.normal(size=(4, 4)), rng.normal(size=(4, 4)), rng.normal(size=(4, 4))
x0 = rng.normal(size=(2, 4))

# Без checkpointing: храним все a1, a2, a3
a1 = layer(x0, W1); a2 = layer(a1, W2); a3 = layer(a2, W3)

# С checkpointing: храним x0 и a3, активацию a1, a2 пересчитаем при backward
# Для иллюстрации просто пересчитаем и сравним:
a1_re = layer(x0, W1); a2_re = layer(a1_re, W2)
print(f"Checkpointing: max diff после recompute = {np.abs(a2 - a2_re).max():.2e}")

# === Gradient accumulation: 4 mini-batch'а по 8 эл-тов накапливаем -> эфф. батч 32 ===
n_total = 32; mb = 8; K = n_total // mb
data = rng.normal(size=(n_total, 3))
labels = rng.normal(size=n_total)
w = np.zeros(3); lr = 0.05

# обычный шаг с батчем 32
g_full = (data * (data @ w - labels)[:, None]).mean(axis=0)

# accumulated шаг
g_acc = np.zeros(3)
for k in range(K):
    s = slice(k * mb, (k + 1) * mb)
    g_acc += (data[s] * (data[s] @ w - labels[s])[:, None]).mean(axis=0) / K
print(f"Accumulation: ||g_full - g_acc|| = {np.linalg.norm(g_full - g_acc):.2e}")
