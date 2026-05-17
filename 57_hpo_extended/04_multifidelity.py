"""Раздел 57 — HPO.

Файл 4: Multi-fidelity HPO.

Концепция: Successive Halving (SHA).
Запускаем N конфигов на малом бюджете r0, оставляем 1/eta лучших, удваиваем бюджет.
Повторяем log_eta(N) раз. Тратит больше ресурсов на перспективные конфиги.

Концепция: Hyperband.
Над SHA: запускает несколько SHA-"скобок" с разными (N, r0) — балансирует
агрессивный отсев против риска "слишком рано выбросить хороший конфиг".

Концепция: ASHA (Asynchronous Successive Halving).
Параллельная асинхронная версия SHA: как только конфиг достигает уровня — продвигаем,
если он в верхних 1/eta среди уже завершивших на этом уровне. Хорошо масштабируется.

Концепция: BOHB.
Hyperband + Bayesian Optimization: вместо random sampling новых конфигов
используется TPE-модель на основе уже виденных результатов.
"""
import numpy as np

rng = np.random.default_rng(0)

# Имитация: качество конфига = функция от "истинного скрытого качества" + шум,
# убывающего с ростом бюджета.
N_init = 27
eta = 3
configs = [{"id": i, "true_score": rng.uniform()} for i in range(N_init)]

def evaluate(cfg, budget):
    noise = rng.normal(scale=1.0 / np.sqrt(budget))
    return cfg["true_score"] + noise

# === Successive Halving ===
alive = configs[:]
budget = 1
while len(alive) > 1:
    scores = [(c, evaluate(c, budget)) for c in alive]
    scores.sort(key=lambda x: -x[1])
    keep = max(1, len(alive) // eta)
    alive = [c for c, _ in scores[:keep]]
    print(f"SHA: budget={budget}, оставили {len(alive)}")
    budget *= eta

best = alive[0]
true_best = max(configs, key=lambda c: c["true_score"])
print(f"SHA финал: выбранный true_score={best['true_score']:.3f}, реально лучший={true_best['true_score']:.3f}")

# === Hyperband: несколько скобок ===
def sha_run(pop, r0, eta=3):
    budget = r0
    alive = pop[:]
    while len(alive) > 1:
        scores = [(c, evaluate(c, budget)) for c in alive]
        scores.sort(key=lambda x: -x[1])
        alive = [c for c, _ in scores[:max(1, len(alive) // eta)]]
        budget *= eta
    return alive[0]

R_max = 27
brackets = []
for s in [3, 2, 1, 0]:
    n = int(np.ceil((R_max / eta ** s) * eta ** s / R_max * eta ** s))
    n = max(n, 2)
    r0 = R_max // (eta ** s) if s > 0 else R_max
    pop = [{"id": i, "true_score": rng.uniform()} for i in range(n)]
    brackets.append(sha_run(pop, r0))
hb_best = max(brackets, key=lambda c: c["true_score"])
print(f"Hyperband: 4 скобки, лучший true_score={hb_best['true_score']:.3f}")

# === ASHA: концепт — двигаем в верхние 1/eta среди завершенных на уровне ===
pop = [{"id": i, "true_score": rng.uniform()} for i in range(20)]
level_scores = {0: [], 1: [], 2: []}
promoted = set()
for c in pop:
    s0 = evaluate(c, 1); level_scores[0].append((c["id"], s0))
    # промоушн на уровень 1 если в топ-1/eta
    if len(level_scores[0]) >= eta:
        thresh = np.quantile([s for _, s in level_scores[0]], 1 - 1.0 / eta)
        if s0 >= thresh and c["id"] not in promoted:
            promoted.add(c["id"])
            level_scores[1].append((c["id"], evaluate(c, 3)))
print(f"ASHA: уровень-1 промоушн={len(level_scores[1])} из {len(pop)}")

# === BOHB концепт: на каждой итерации SHA новые конфиги сэмплируем из TPE-модели ===
print("BOHB: Hyperband-скобки + TPE-предложение конфигов вместо random — формула как в TPE.")
