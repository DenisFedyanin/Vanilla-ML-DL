"""Раздел 36 — RNN. Файл 2: Bidirectional и Stacked (многослойные) RNN.

Концепция: Bidirectional RNN.
Прогоняем последовательность вперёд (h_fw) и назад (h_bw),
конкатенируем по последней оси: h_t = [h_fw_t; h_bw_t].
Каждый момент времени видит и прошлое, и будущее.
Применение: POS-теггинг, NER, BiLSTM как кодировщик.

Концепция: Stacked / multi-layer RNN.
Несколько слоёв: вход следующего слоя = выходная последовательность предыдущего.
Глубина увеличивает выразительность. Часто 2-4 слоя; с dropout между.

Концепция: residual connection между слоями RNN.
В глубоких RNN добавляют h^(l+1) = RNN(h^l) + h^l (если размерности совпадают).
Помогает обучению, как и в CNN.
"""
import numpy as np

rng = np.random.default_rng(0)


def rnn_forward(x_seq, Wx, Wh, b):
    """Возвращает (T, d_h) — последовательность скрытых состояний."""
    d_h = Wh.shape[0]
    h = np.zeros(d_h)
    out = []
    for t in range(x_seq.shape[0]):
        h = np.tanh(x_seq[t] @ Wx + h @ Wh + b)
        out.append(h.copy())
    return np.stack(out)


# === Bidirectional RNN ===
print("=== Bidirectional RNN ===")
T, d_in, d_h = 6, 4, 3
x_seq = rng.standard_normal((T, d_in))

Wx_f = rng.standard_normal((d_in, d_h)) * 0.3
Wh_f = rng.standard_normal((d_h, d_h)) * 0.3
Wx_b = rng.standard_normal((d_in, d_h)) * 0.3
Wh_b = rng.standard_normal((d_h, d_h)) * 0.3
b_f = np.zeros(d_h)
b_b = np.zeros(d_h)

h_fw = rnn_forward(x_seq, Wx_f, Wh_f, b_f)
h_bw = rnn_forward(x_seq[::-1], Wx_b, Wh_b, b_b)[::-1]
h_bi = np.concatenate([h_fw, h_bw], axis=-1)
print(f"x{x_seq.shape}; h_fw{h_fw.shape}, h_bw{h_bw.shape}, конкат -> h_bi{h_bi.shape}")
print(f"h_t в момент t=2: первые {d_h} от прошлого, последние {d_h} от будущего")

# === Stacked RNN: 3 слоя ===
print("\n=== Stacked RNN (3 слоя) ===")
seq = x_seq
for layer in range(3):
    Wx_l = rng.standard_normal((seq.shape[1], d_h)) * 0.3
    Wh_l = rng.standard_normal((d_h, d_h)) * 0.3
    out = rnn_forward(seq, Wx_l, Wh_l, np.zeros(d_h))
    print(f"layer{layer+1}: in{seq.shape} -> out{out.shape}")
    seq = out

# === Residual stacked: y = RNN(x) + x (если размерности совпадают) ===
print("\n=== Residual stacked RNN ===")
seq = rng.standard_normal((T, d_h))
norm_before = np.linalg.norm(seq)
Wx_l = rng.standard_normal((d_h, d_h)) * 0.3
Wh_l = rng.standard_normal((d_h, d_h)) * 0.3
out = rnn_forward(seq, Wx_l, Wh_l, np.zeros(d_h)) + seq
print(f"||in||={norm_before:.2f}, ||out||={np.linalg.norm(out):.2f}")
print("Residual сохраняет масштаб и градиенты при глубоких stack")
