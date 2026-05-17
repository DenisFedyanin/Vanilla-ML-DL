"""Раздел 41 — Предобученные LM. Файл 1: цели обучения и метрики.

Концепция: Autoregressive LM (GPT).
Моделируем p(x) = prod p(x_t | x_<t). Декодер с causal mask.
Лосс — среднее cross-entropy по позициям.

Концепция: Perplexity.
PPL = exp(средний NLL по токенам) = exp(loss). Чем меньше — тем модель лучше.
Интуиция: эффективное число равновероятных продолжений.

Концепция: MLM (BERT).
Маскируем 15% токенов, предсказываем оригинал. Двусторонний контекст.

Концепция: NSP (Next Sentence Prediction).
Бинарная задача "являются ли A и B соседями?". В BERT, оригинально, потом отказались.

Концепция: SOP (Sentence Order Prediction, ALBERT).
Усиление NSP: реальный порядок vs перевёрнутый. Сложнее, информативнее.
"""
import numpy as np

rng = np.random.default_rng(0)

# === Perplexity на toy токенах ===
V = 5
T = 8
seq = rng.integers(0, V, size=T)
# Имитируем какое-то предсказательное распределение
def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


logits = rng.standard_normal((T, V))
probs = softmax(logits, axis=-1)
# Cross-entropy на истинных токенах
nll = -np.log(probs[np.arange(T), seq] + 1e-9)
loss = nll.mean()
ppl = np.exp(loss)
print(f"=== Causal LM PPL ===")
print(f"sequence: {seq.tolist()}")
print(f"средний NLL = {loss:.3f}, PPL = exp(loss) = {ppl:.2f}")
print(f"Случайная модель: PPL ≈ {V} (равномерно)")

# Идеальная модель (правильный токен -> p=1)
print(f"Идеальная (p=1 на верный токен): PPL = 1.0")

# === MLM пример ===
print("\n=== MLM ===")
mask = rng.uniform(size=T) < 0.15
print(f"маска: {mask.astype(int).tolist()}")
print(f"входы для модели: {[101 if m else int(s) for s, m in zip(seq, mask)]}  (101 = [MASK])")
print(f"таргеты только в позициях маски: {seq[mask].tolist()}")
loss_mlm = -np.log(probs[mask, seq[mask]] + 1e-9).mean() if mask.any() else 0
print(f"MLM loss = {loss_mlm:.3f}")

# === NSP / SOP ===
print("\n=== NSP / SOP ===")
print("NSP: P(IsNext | [CLS] A [SEP] B [SEP]) бинарная классификация.")
print("    50% - реальные соседи, 50% - случайные.")
print("SOP: 50% - правильный порядок (A,B), 50% - перевёрнутый (B,A).")
print("    Сложнее NSP, заставляет учить связность.")
