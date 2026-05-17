"""Раздел 62 — Трюки обучения.

Файл 5: Сжатие моделей.

Концепция: Knowledge distillation.
Студент учится не на жестких labels, а на softmax-распределении учителя:
L = (1-a) CE(y, p_s) + a * T^2 KL(softmax(z_t/T) || softmax(z_s/T)).
T (temperature) "размывает" softmax. Передает "темные знания" учителя.

Концепция: Magnitude pruning.
Зануляем долю k% весов с наименьшим |w|. Обычно с iterative pruning и fine-tuning.
Дает sparse модели — экономия памяти/вычислений.

Концепция: Structured pruning.
Удаляем целые структуры: нейроны, каналы, головы. В отличие от magnitude дает
реальное ускорение на стандартном hardware (нет sparse-формата).

Концепция: PTQ (Post-Training Quantization).
Без переобучения: w_q = round(w/scale)+zero, dequant: w ≈ scale*(w_q - zero).
INT8 -> 4x экономия памяти, быстрые INT-операции.

Концепция: QAT (Quantization-Aware Training).
Во время обучения симулируем округление в forward, gradient идет через "straight-through estimator".
Качество выше PTQ при той же битности.
"""
import numpy as np

rng = np.random.default_rng(0)

# === Distillation: KL между student/teacher логитами с температурой ===
z_t = np.array([2.0, 1.0, 0.1])  # teacher logits
z_s = np.array([1.5, 1.2, 0.3])  # student
T = 2.0
softmax = lambda z: np.exp(z - z.max()) / np.exp(z - z.max()).sum()
p_t = softmax(z_t / T); p_s = softmax(z_s / T)
KL = np.sum(p_t * np.log((p_t + 1e-12) / (p_s + 1e-12)))
distill_loss = T * T * KL
print(f"Distillation: KL={KL:.4f}, distill_loss(T={T})={distill_loss:.4f}")

# === Magnitude pruning ===
W = rng.normal(size=(4, 5))
k = 0.5  # обнулить 50%
mask = np.abs(W) > np.quantile(np.abs(W), k)
W_pruned = W * mask
print(f"До prune: nnz={W.size}, после prune: nnz={mask.sum()} ({mask.mean()*100:.0f}%)")
print(f"Сохранили только большие по |.| веса")

# === Structured pruning: выкинем самый "слабый" канал (строку) по ||row||_2 ===
norms = np.linalg.norm(W, axis=1)
weakest = norms.argmin()
W_struct = np.delete(W, weakest, axis=0)
print(f"Structured: убрали строку {weakest} с нормой {norms[weakest]:.3f}, форма {W.shape} -> {W_struct.shape}")

# === PTQ INT8: симметричное квантование ===
def quantize_dequantize(w, bits=8):
    qmax = 2 ** (bits - 1) - 1  # 127
    scale = np.abs(w).max() / qmax
    q = np.clip(np.round(w / scale), -qmax - 1, qmax).astype(np.int8)
    return q, scale, q.astype(float) * scale

q, scale, w_dq = quantize_dequantize(W)
err = np.abs(W - w_dq).max()
print(f"PTQ INT8: scale={scale:.4f}, max abs err={err:.4f}, относ. ~{err/np.abs(W).max()*100:.1f}%")

# === QAT: forward с fake-quant, grad через STE ===
# Пример: сделаем 1 шаг "обучения" где в forward используется dequant, а grad протекает к w
w = np.array([0.5, 1.3, -0.7])
target = np.array([0.4, 1.0, -0.6])
_, _, w_fake = quantize_dequantize(w, bits=4)
loss = 0.5 * ((w_fake - target) ** 2).sum()
g = w_fake - target  # STE: grad протекает прямо как от identity
w_new = w - 0.1 * g
print(f"QAT-style: w={w}, после шага: w={w_new.round(3)}")
