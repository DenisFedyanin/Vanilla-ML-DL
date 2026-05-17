"""Раздел 30 — Оптимизаторы (расширенно). Файл 4: мета-оптимизаторы.

Концепция 338: Lookahead.
Держим "медленные" веса slow и "быстрые" fast. K шагов обычным оптимизатором
по fast, затем slow := slow + alpha*(fast-slow), fast := slow. Стабилизирует.

Концепция 339: SWA (Stochastic Weight Averaging).
Усредняем веса по последним эпохам обучения. На том же лоссе обобщает лучше,
часто даёт +0.5..1.5% точности.

Концепция 340: Polyak averaging (EMA of weights).
w_ema = beta * w_ema + (1-beta) * w. Сглаживает траекторию,
используется в EMA-моделях (diffusion, обнаружение).

Концепция 341: Gradient accumulation.
Не делаем шаг каждый мини-батч. Накапливаем градиент по K мини-батчам,
делим на K, делаем один шаг. Имитирует большой батч с малой памятью.
"""
import numpy as np

rng = np.random.default_rng(0)

# Задача: min f(x) = 0.5 ||x - target||^2
target = np.array([3.0, -2.0])
def grad(x): return x - target
def f(x): return 0.5 * np.sum((x - target) ** 2)

# === 338: Lookahead на базе SGD ===
slow = np.zeros(2); fast = slow.copy()
alpha, K, lr = 0.5, 5, 0.1
for outer in range(10):
    for _ in range(K):
        fast = fast - lr * grad(fast)
    slow = slow + alpha * (fast - slow)
    fast = slow.copy()
print(f"338 Lookahead: slow={slow.round(4).tolist()}, f={f(slow):.5f}")

# === 339: SWA ===
x = np.zeros(2)
snapshots = []
for t in range(40):
    x = x - 0.1 * grad(x)
    if t >= 20:                        # начинаем усреднять во второй половине
        snapshots.append(x.copy())
swa = np.mean(snapshots, axis=0)
print(f"339 SWA: финальный x={x.round(4).tolist()}, SWA-avg={swa.round(4).tolist()}, f_swa={f(swa):.5f}")

# === 340: Polyak EMA ===
x = np.zeros(2); ema = np.zeros(2); beta = 0.95
for _ in range(50):
    x = x - 0.1 * grad(x)
    ema = beta * ema + (1 - beta) * x
print(f"340 EMA(beta={beta}): x={x.round(4).tolist()}, ema={ema.round(4).tolist()}, f_ema={f(ema):.5f}")

# === 341: Gradient accumulation ===
# Имитируем стохастический градиент: к точному g добавляем шум по мини-батчу.
x = np.zeros(2); K_acc, lr = 4, 0.1
for step in range(50):
    g_sum = np.zeros(2)
    for k in range(K_acc):
        noise = rng.standard_normal(2) * 0.5
        g_sum += grad(x) + noise       # шумные мини-батчи
    g_avg = g_sum / K_acc
    x = x - lr * g_avg
print(f"341 GradAccum K={K_acc}: x={x.round(4).tolist()}, f={f(x):.5f}")
print("    Имитирует большой батч (меньше шум) при малой памяти.")
