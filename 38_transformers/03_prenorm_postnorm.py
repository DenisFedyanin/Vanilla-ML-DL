"""Раздел 38 — Трансформеры. Файл 3: Pre-LN vs Post-LN.

Концепция: Post-LN (оригинальный, Vaswani 2017).
y = LN(x + sublayer(x)). LN ПОСЛЕ residual.
Без warmup'a часто расходится; чувствителен к learning rate.

Концепция: Pre-LN (современный, GPT-2 и далее).
y = x + sublayer(LN(x)). LN ПЕРЕД sublayer.
Стабильнее обучается, можно без долгого warmup, лучше масштабируется по глубине.

Концепция: эффект на масштаб активаций.
В Post-LN норма выхода нормализована LN: ограничена.
В Pre-LN residual путь свободно копится, но не взрывается, так как sublayer на нормированном входе.
"""
import numpy as np

rng = np.random.default_rng(0)


def layer_norm(x, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def sublayer(x, W):
    """Простой 'sublayer' — линейная трансформация (имитация FFN/Attention)."""
    return np.maximum(0, x @ W)


T, d, depth = 5, 8, 6
x = rng.standard_normal((T, d))
Ws = [rng.standard_normal((d, d)) * 0.4 for _ in range(depth)]

# === Post-LN ===
x_post = x.copy()
norms_post = [np.linalg.norm(x_post)]
for W in Ws:
    x_post = layer_norm(x_post + sublayer(x_post, W))
    norms_post.append(np.linalg.norm(x_post))

# === Pre-LN ===
x_pre = x.copy()
norms_pre = [np.linalg.norm(x_pre)]
for W in Ws:
    x_pre = x_pre + sublayer(layer_norm(x_pre), W)
    norms_pre.append(np.linalg.norm(x_pre))

print("=== Норма активаций по слоям ===")
print(f"Post-LN: {[round(n, 2) for n in norms_post]}")
print(f"Pre-LN:  {[round(n, 2) for n in norms_pre]}")
print()
print("Post-LN: норма после каждого слоя ~ const (LN нормирует выход).")
print("Pre-LN:  норма растёт (residual путь без финальной LN), но обучение стабильнее.")
print()
print("Практика: Pre-LN — стандарт в современных LLM (GPT-2/3, LLaMA).")
