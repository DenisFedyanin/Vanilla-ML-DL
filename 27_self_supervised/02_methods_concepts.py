"""Раздел 27 — Self-supervised learning. Файл 2: ключевые методы (концепты).

Концепция 707: SimCLR.
Большие батчи, две аугментации, InfoNCE по батчу, projection head g(.) поверх
бэкбона f(.). Контраст в пространстве g(f(x)), но downstream использует f(x).
Главный приём — сильные аугментации и большой batch для богатого negatives.

Концепция 708: MoCo (Momentum Contrast).
Очередь negatives (queue) + momentum-encoder, чьи веса — EMA весов основного.
Так получают «много» negatives без огромного батча. Контраст по cosine sim.

Концепция 709: BYOL (Bootstrap Your Own Latent).
БЕЗ negatives. Online-сеть учится предсказывать выход target-сети (EMA online).
Стабильность обеспечивается асимметрией: predictor поверх online + stop-gradient
на target. Loss = MSE между нормализованными предиктом и таргетом.

Концепция 710: SimSiam.
То же, что BYOL, но БЕЗ EMA. Просто две одинаковые сети, predictor и stop-gradient.
Удивительно, не коллапсирует — ключ в stop-gradient.

Концепция 711: Barlow Twins.
Loss работает с кросс-корреляционной матрицей C двух представлений (после
batch-norm). Цель: C ≈ I — диагональ к 1 (инвариантность), вне-диагональ к 0
(декорреляция). Не нужны negatives и stop-gradient.
"""
import numpy as np

# Печатаем pseudo-code каждого метода.
print("707 SimCLR (псевдо-код):")
for line in [
    "x1, x2 = aug(x), aug(x)",
    "z1, z2 = g(f(x1)), g(f(x2))",
    "loss = InfoNCE(z1, z2, batch_negatives)",
]:
    print(f"     {line}")

print("708 MoCo (псевдо-код):")
for line in [
    "k = momentum_encoder(x_key)        # ema of f",
    "q = f(x_query)",
    "logits = [q.k, q.queue]; labels = 0",
    "loss = CE(logits/T, labels); queue.enqueue(k); queue.dequeue()",
]:
    print(f"     {line}")

print("709 BYOL (псевдо-код):")
for line in [
    "online: y = f(x); z = g(y); p = q(z)   # predictor",
    "target: y' = f'(x); z' = g'(y')        # f', g' = EMA(f, g)",
    "loss = MSE(normalize(p), normalize(stop_grad(z')))",
    "ema_update(f', f); ema_update(g', g)",
]:
    print(f"     {line}")

print("710 SimSiam (псевдо-код):")
for line in [
    "z1, z2 = f(x1), f(x2)",
    "p1, p2 = h(z1), h(z2)            # predictor",
    "loss = -0.5*(cos(p1, stop_grad(z2)) + cos(p2, stop_grad(z1)))",
]:
    print(f"     {line}")

print("711 Barlow Twins (псевдо-код):")
for line in [
    "z1, z2 = BN(f(x1)), BN(f(x2))",
    "C = (z1.T @ z2) / N",
    "loss = sum_i (1 - C_ii)^2 + lambda * sum_{i!=j} C_ij^2",
]:
    print(f"     {line}")

# Микро-демо Barlow-Twins-loss на случайных представлениях.
rng = np.random.default_rng(0)
N, D = 64, 8
def bn(z):
    return (z - z.mean(0)) / (z.std(0) + 1e-12)
z1 = bn(rng.normal(size=(N, D)))
z2 = bn(rng.normal(size=(N, D)))
C = (z1.T @ z2) / N
off_diag = (C ** 2).sum() - (np.diag(C) ** 2).sum()
diag_term = ((1 - np.diag(C)) ** 2).sum()
print(f"711 Barlow-Twins loss на случайных z: diag_term={diag_term:.2f}, "
      f"off_diag_pen={off_diag:.2f}")
