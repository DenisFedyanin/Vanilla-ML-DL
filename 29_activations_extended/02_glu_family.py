"""Раздел 29 — Активации (расширенно). Файл 2: Maxout и GLU-семейство.

Концепция 313: Maxout.
Берём k линейных проекций x->W_i x + b_i и поэлементный max:
y = max_i (W_i x + b_i). Универсальная аппроксимация выпуклых функций.

Концепция 314: GLU (Gated Linear Unit).
Разделяем выход полносвязного на две половины (a, b),
y = a * sigmoid(b). Половина "контент", половина "гейт".
Используется в LLM-FFN (PaLM, LLaMA).

Концепция 315: ReGLU = a * ReLU(b).
Вариант GLU с ReLU вместо sigmoid в гейте.

Концепция 316: SwiGLU = a * SiLU(b) = a * (b * sigmoid(b)).
Лучшая активация в FFN современных LLM (LLaMA, Mistral).

Концепция 317: GeGLU = a * GELU(b).
GLU с GELU. Использовалась в T5/PaLM-flavoured моделях.
"""
import numpy as np

rng = np.random.default_rng(0)

# === 313: Maxout (k=3 линейных юнита) ===
def maxout(x, Ws, bs):
    # x: (B, in); Ws: list of (in, out); bs: list of (out,)
    outs = [x @ W + b for W, b in zip(Ws, bs)]
    return np.maximum.reduce(outs)

B, in_f, out_f, k = 3, 4, 2, 3
x = rng.standard_normal((B, in_f))
Ws = [rng.standard_normal((in_f, out_f)) * 0.5 for _ in range(k)]
bs = [np.zeros(out_f) for _ in range(k)]
y_mo = maxout(x, Ws, bs)
print(f"313 Maxout k={k}: x{x.shape} -> y{y_mo.shape}")
print(f"    каждый выход = max из {k} линейных проекций")

# Общее тело для GLU-семейства: вход (B, 2*d) -> расщепляется на a, b
d = 4
h = rng.standard_normal((B, 2 * d))
a, b = h[:, :d], h[:, d:]

def sigmoid(x): return 1 / (1 + np.exp(-x))
def silu(x): return x * sigmoid(x)
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

# === 314: GLU ===
y_glu = a * sigmoid(b)
print(f"314 GLU: in{h.shape} -> out{y_glu.shape} (gate=sigmoid(b))")

# === 315: ReGLU ===
y_re = a * np.maximum(0, b)
print(f"315 ReGLU: out{y_re.shape}, mean={y_re.mean():.3f}")

# === 316: SwiGLU ===
y_sw = a * silu(b)
print(f"316 SwiGLU: out{y_sw.shape}, mean={y_sw.mean():.3f} — стандарт LLaMA/Mistral")

# === 317: GeGLU ===
y_ge = a * gelu(b)
print(f"317 GeGLU: out{y_ge.shape}, mean={y_ge.mean():.3f}")

# Сравним L2-нормы по батчу
for name, y in [("GLU", y_glu), ("ReGLU", y_re), ("SwiGLU", y_sw), ("GeGLU", y_ge)]:
    print(f"   ||{name}||_F = {np.linalg.norm(y):.3f}")
