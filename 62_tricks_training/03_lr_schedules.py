"""Раздел 62 — Трюки обучения.

Файл 3: Расписания learning rate.

Концепция: Warmup.
Несколько эпох/шагов lr линейно растет от 0 (или маленького) до base_lr.
Помогает большим моделям/большим батчам в начале обучения.

Концепция: Constant.
lr = base. Базовая опция, проста и предсказуема.

Концепция: Step decay.
lr = base * gamma^(t // step_size). Падает скачками на gamma каждые step_size шагов.

Концепция: Exponential decay.
lr = base * gamma^t. Плавное экспоненциальное затухание.

Концепция: Cosine annealing.
lr = lr_min + 0.5 (base - lr_min) (1 + cos(pi t / T)). Плавно от base к lr_min.

Концепция: Cosine with restarts (SGDR).
То же, но "перезапуски" через каждые T_i шагов, обычно T_{i+1}=2 T_i.

Концепция: ReduceLROnPlateau.
Снижаем lr * factor если loss не улучшался patience эпох. Адаптивно к практике.
"""
import numpy as np

T = 50
base = 0.1
lr_min = 1e-3

# === Warmup на 5 шагов, потом constant ===
warmup = lambda t: base * t / 5 if t < 5 else base
sched_warmup = [warmup(t) for t in range(T)]

# === Constant ===
sched_const = [base for _ in range(T)]

# === Step decay: каждые 15 шагов *0.5 ===
sched_step = [base * 0.5 ** (t // 15) for t in range(T)]

# === Exponential: gamma=0.95 ===
sched_exp = [base * 0.95 ** t for t in range(T)]

# === Cosine annealing ===
sched_cos = [lr_min + 0.5 * (base - lr_min) * (1 + np.cos(np.pi * t / T)) for t in range(T)]

# === Cosine with restarts T_0=10, mult=2 -> 10,20 ===
sched_sgdr = []
period = 10; start = 0
for t in range(T):
    pos = t - start
    sched_sgdr.append(lr_min + 0.5 * (base - lr_min) * (1 + np.cos(np.pi * pos / period)))
    if pos >= period - 1:
        start = t + 1; period *= 2

# === Plateau: имитация — если "loss" не улучшается, делим lr на 2 ===
losses = [1.0, 0.9, 0.85, 0.85, 0.85, 0.8, 0.8, 0.8, 0.8]  # имитация
lr_p = base; best = float('inf'); patience = 2; bad = 0; sched_plateau = []
for L in losses * 6:
    if L < best - 1e-4: best = L; bad = 0
    else: bad += 1
    if bad >= patience: lr_p *= 0.5; bad = 0
    sched_plateau.append(lr_p)

print("Step 0:    ", [f"{s[0]:.4f}" for s in [sched_warmup, sched_const, sched_step, sched_exp, sched_cos, sched_sgdr]])
print("Step 10:   ", [f"{s[10]:.4f}" for s in [sched_warmup, sched_const, sched_step, sched_exp, sched_cos, sched_sgdr]])
print("Step 30:   ", [f"{s[30]:.4f}" for s in [sched_warmup, sched_const, sched_step, sched_exp, sched_cos, sched_sgdr]])
print("Step 49:   ", [f"{s[49]:.4f}" for s in [sched_warmup, sched_const, sched_step, sched_exp, sched_cos, sched_sgdr]])
print(f"Plateau финал: lr={sched_plateau[-1]:.5f} (стартовый {base})")
