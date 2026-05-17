"""Раздел 38 — Трансформеры. Файл 5: цели предобучения.

Концепция: Causal LM (GPT).
Предсказываем следующий токен p(x_t | x_<t). Декодер с causal маской.
Loss = cross-entropy на сдвинутой на 1 последовательности.

Концепция: Masked LM (BERT).
15% случайных позиций заменяются на [MASK]; модель предсказывает оригинал.
Двунаправленный контекст (encoder).

Концепция: Span corruption (T5).
Заменяем целые промежутки на маркеры <X>, <Y>...; целевая — последовательность пропущенных кусков.

Концепция: Replaced token detection (ELECTRA).
Маленький генератор подменяет токены; основная модель — дискриминатор:
для каждого токена бинарно решает, заменён он или нет. Эффективнее MLM на токен.

Концепция: Denoising (BART).
Применяем разные шумы (маскирование, удаление, перестановка, замена спанов)
к входу; модель восстанавливает исходный текст. Encoder-Decoder.
"""
import numpy as np

rng = np.random.default_rng(0)
V = 10
T = 12
seq = rng.integers(2, V, size=T)
print(f"Исходная последовательность: {seq.tolist()}")

# === Causal LM ===
print("\n=== Causal LM (GPT) ===")
inputs = seq[:-1]
targets = seq[1:]
print(f"inputs={inputs.tolist()}")
print(f"targets={targets.tolist()} (сдвиг на 1 вправо)")

# === MLM (BERT) ===
print("\n=== Masked LM (BERT) ===")
MASK_ID = 1
mask = rng.uniform(size=T) < 0.15
masked = seq.copy()
masked[mask] = MASK_ID
print(f"маска={mask.astype(int).tolist()}")
print(f"masked input={masked.tolist()}")
print(f"target = оригинал в позициях маски: {seq[mask].tolist()}")

# === Span corruption (T5) ===
print("\n=== Span corruption (T5) ===")
SENTINEL_BASE = 90    # <X>=90, <Y>=91, ...
out_in = []
out_tgt = []
i = 0
sentinel = 0
while i < T:
    if rng.uniform() < 0.2 and i + 1 < T:
        # начинаем span
        span_len = min(rng.integers(2, 4), T - i)
        out_in.append(SENTINEL_BASE + sentinel)
        out_tgt.append(SENTINEL_BASE + sentinel)
        out_tgt.extend(seq[i:i + span_len].tolist())
        i += span_len
        sentinel += 1
    else:
        out_in.append(int(seq[i]))
        i += 1
out_tgt.append(SENTINEL_BASE + sentinel)
print(f"input ={out_in}")
print(f"target={out_tgt}")

# === Replaced token detection (ELECTRA) ===
print("\n=== ELECTRA replaced token detection ===")
replaced = seq.copy()
labels = np.zeros(T, dtype=int)
for t in range(T):
    if rng.uniform() < 0.15:
        replaced[t] = rng.integers(2, V)
        labels[t] = 1 if replaced[t] != seq[t] else 0
print(f"replaced={replaced.tolist()}")
print(f"labels (1=заменён)={labels.tolist()}")

# === Denoising (BART) ===
print("\n=== BART denoising (token deletion + mask span) ===")
noisy = seq.copy().tolist()
# token deletion
keep = rng.uniform(size=T) > 0.1
noisy_del = [int(t) for t, k in zip(noisy, keep) if k]
print(f"после deletion: {noisy_del}")
# затем добавим маск-спан
ins = rng.integers(1, len(noisy_del))
noisy_final = noisy_del[:ins] + [MASK_ID] + noisy_del[ins + 2:]
print(f"+ mask span: {noisy_final}")
print(f"target = оригинал: {seq.tolist()}")
