"""Раздел 36 — RNN расширение. Файл 1: варианты рекуррентных ячеек.

Концепция: Vanilla RNN.
h_t = tanh(W_x x_t + W_h h_{t-1} + b). Простейшая рекуррентность.
Страдает от затухания/взрыва градиентов; не работает на длинных зависимостях.

Концепция: LSTM (стандартный).
4 гейта: forget, input, cell-candidate, output. Поддерживает cell state c_t.
c_t = f*c_{t-1} + i*g; h_t = o * tanh(c_t).
Решает проблему long-term dependencies.

Концепция: Peephole LSTM.
Гейты видят также cell state c_{t-1}. Иногда улучшает точность таймингов.
Формула: f = σ(W_f [x_t, h_{t-1}, c_{t-1}] + b_f) и аналогично для i, o.

Концепция: GRU.
Объединяет cell и hidden state. 2 гейта: update z, reset r.
h_t = (1-z)*h_{t-1} + z*tanh(W_x x + W_h (r * h_{t-1})).
Быстрее, чем LSTM, часто близкого качества.

Концепция: MGU (Minimal Gated Unit).
Дальнейшее упрощение: один гейт f. Меньше параметров, сравнимая точность.
h_t = (1-f)*h_{t-1} + f*tanh(W_x x + W_h (f*h_{t-1})).
"""
import numpy as np

rng = np.random.default_rng(0)


def tanh(x):
    return np.tanh(x)


# === Vanilla RNN forward ===
print("=== Vanilla RNN ===")
T, d_in, d_h = 5, 3, 4
Wx = rng.standard_normal((d_in, d_h)) * 0.3
Wh = rng.standard_normal((d_h, d_h)) * 0.3
b = np.zeros(d_h)
x_seq = rng.standard_normal((T, d_in))
h = np.zeros(d_h)
hs = []
for t in range(T):
    h = tanh(x_seq[t] @ Wx + h @ Wh + b)
    hs.append(h.copy())
hs = np.stack(hs)
print(f"x_seq{x_seq.shape} -> h_seq{hs.shape}")
print(f"||h_t|| по шагам: {[round(np.linalg.norm(hh), 3) for hh in hs]}")

# === LSTM (один шаг) — для иллюстрации ===
print("\n=== LSTM (один шаг) ===")
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x_t = rng.standard_normal(d_in)
h_prev = np.zeros(d_h)
c_prev = np.zeros(d_h)
# объединённая матрица для 4 гейтов
W = rng.standard_normal((d_in + d_h, 4 * d_h)) * 0.3
z = np.concatenate([x_t, h_prev]) @ W
i, f, g, o = np.split(z, 4)
i, f, o = sigmoid(i), sigmoid(f), sigmoid(o)
g = tanh(g)
c = f * c_prev + i * g
h = o * tanh(c)
print(f"gates: i.mean={i.mean():.2f}, f.mean={f.mean():.2f}, o.mean={o.mean():.2f}")
print(f"c{c.shape}, h{h.shape}")

# === Peephole — добавим c_prev в вычисление f, i, o ===
print("\nPeephole: f = σ(W_f [x, h_prev, c_prev]) — гейты видят c напрямую")

# === GRU (один шаг) ===
print("\n=== GRU (один шаг) ===")
Wz = rng.standard_normal((d_in + d_h, d_h)) * 0.3
Wr = rng.standard_normal((d_in + d_h, d_h)) * 0.3
Wn = rng.standard_normal((d_in + d_h, d_h)) * 0.3
z_g = sigmoid(np.concatenate([x_t, h_prev]) @ Wz)
r_g = sigmoid(np.concatenate([x_t, h_prev]) @ Wr)
n_g = tanh(np.concatenate([x_t, r_g * h_prev]) @ Wn)
h_gru = (1 - z_g) * h_prev + z_g * n_g
print(f"update z.mean={z_g.mean():.2f}, reset r.mean={r_g.mean():.2f}, h{h_gru.shape}")

# === MGU (один гейт) ===
print("\n=== MGU ===")
Wf = rng.standard_normal((d_in + d_h, d_h)) * 0.3
Wn_mgu = rng.standard_normal((d_in + d_h, d_h)) * 0.3
f_g = sigmoid(np.concatenate([x_t, h_prev]) @ Wf)
n_g = tanh(np.concatenate([x_t, f_g * h_prev]) @ Wn_mgu)
h_mgu = (1 - f_g) * h_prev + f_g * n_g
print(f"MGU единственный гейт f.mean={f_g.mean():.2f}, h{h_mgu.shape}")
