"""Раздел 39 — NLP. Файл 5: разметка последовательностей.

Концепция: POS-tagging.
Каждому слову — тег части речи (NOUN, VERB, ADJ, ...). Базовая задача NLP.

Концепция: HMM (Hidden Markov Model).
Скрытые состояния = теги, наблюдения = слова.
Параметры: π (начальные), A (transition tag->tag), B (emission tag->word).

Концепция: алгоритм Витерби.
Динамическое программирование: V_t(j) = max_i V_{t-1}(i) * A[i,j] * B[j, w_t].
Возвращает наиболее вероятную последовательность тегов за O(T*K^2).

Концепция: CRF (Conditional Random Field).
Дискриминативная модель: моделирует P(теги | слова) напрямую,
учитывает признаки соседних меток. Поверх любых признаков (включая нейронные).
"""
import numpy as np

# === Tiny HMM ===
# 2 тега: 0=NOUN, 1=VERB; словарь {fish, run, swim, cat}
tags = ["NOUN", "VERB"]
vocab = ["fish", "run", "swim", "cat"]
v_idx = {w: i for i, w in enumerate(vocab)}

# Параметры HMM (заданы вручную, для иллюстрации)
pi = np.array([0.7, 0.3])                       # обычно начинаем с NOUN
A = np.array([[0.4, 0.6],                       # NOUN -> NOUN/VERB
              [0.7, 0.3]])                      # VERB -> NOUN/VERB
B = np.array([[0.4, 0.05, 0.05, 0.5],           # NOUN: fish/run/swim/cat
              [0.2, 0.4,  0.35, 0.05]])         # VERB: fish/run/swim/cat

sentence = ["fish", "swim"]
obs = [v_idx[w] for w in sentence]
T = len(obs)
K = len(tags)

# === Алгоритм Витерби ===
V = np.zeros((T, K))
back = np.zeros((T, K), dtype=int)
V[0] = pi * B[:, obs[0]]
for t in range(1, T):
    for j in range(K):
        scores = V[t - 1] * A[:, j] * B[j, obs[t]]
        back[t, j] = int(np.argmax(scores))
        V[t, j] = scores[back[t, j]]

# Восстановление пути
best = [int(np.argmax(V[-1]))]
for t in range(T - 1, 0, -1):
    best.append(back[t, best[-1]])
best = best[::-1]
print("=== Viterbi на HMM ===")
print(f"предложение: {sentence}")
print(f"V[T]: {np.round(V, 4)}")
print(f"лучшая разметка: {[tags[i] for i in best]}")
print(f"вероятность последовательности: {V[-1].max():.4f}")

# === CRF — концепция ===
print("\n=== CRF (концепция) ===")
print("CRF моделирует P(y|x) = exp(sum_t [emission_t(y_t) + transition(y_{t-1}, y_t)]) / Z(x).")
print("Параметры обучаются по condit. likelihood. Вывод — также Витерби.")
print("В отличие от HMM, признаки x могут быть любыми (контекстными, char-gram, BiLSTM-выход).")
