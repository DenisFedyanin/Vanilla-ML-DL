"""Раздел 29 — Активации (расширенно). Файл 3: выходные активации.

Концепция 318: Sigmoid.
sigmoid(x) = 1/(1+exp(-x)). Сжимает в (0,1). Используется в бинарной классификации
и как гейт. Минусы: насыщается, градиент -> 0.

Концепция 319: Softmax с трюком стабильности.
softmax(x)_i = exp(x_i) / sum_j exp(x_j). Чтобы избежать overflow,
вычитаем max(x) — результат не меняется, но числа маленькие.

Концепция 320: Log-Softmax.
log(softmax(x)) = x - logsumexp(x). Численно стабильнее, чем log(softmax(x)) в лоб.
Применяется в NLL-лоссе.

Концепция 321: Sparsemax (концепт).
Альтернатива softmax: возвращает разрежённое распределение (часть p_i = 0).
Полезна в attention для интерпретируемости.

Концепция 322: Hierarchical softmax (концепт).
Дерево классов; вероятность класса — произведение бинарных решений
по пути от корня. Сложность O(log V) вместо O(V). Используется в word2vec.
"""
import numpy as np

# === 318: Sigmoid ===
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

logits = np.array([-2.0, 0.0, 2.0])
print(f"318 Sigmoid({logits.tolist()}) = {sigmoid(logits).round(3).tolist()}")

# === 319: Softmax (стабильный) ===
def softmax(x):
    s = x - x.max(axis=-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=-1, keepdims=True)

x_big = np.array([1000.0, 1001.0, 1002.0])
p = softmax(x_big)
print(f"319 Softmax([1000,1001,1002]) = {p.round(3).tolist()} (без трюка был бы overflow)")

# === 320: Log-softmax через logsumexp ===
def logsumexp(x):
    m = x.max(axis=-1, keepdims=True)
    return m.squeeze(-1) + np.log(np.exp(x - m).sum(axis=-1))

def log_softmax(x):
    return x - logsumexp(x)[..., None]

ls = log_softmax(x_big)
print(f"320 LogSoftmax = {ls.round(3).tolist()}, exp(ls) = {np.exp(ls).round(3).tolist()}")

# === 321: Sparsemax (через сортировку) ===
def sparsemax(z):
    z_sorted = np.sort(z)[::-1]
    k = np.arange(1, len(z) + 1)
    cond = 1 + k * z_sorted > np.cumsum(z_sorted)
    k_star = k[cond][-1]
    tau = (np.cumsum(z_sorted)[k_star - 1] - 1) / k_star
    return np.maximum(z - tau, 0)

z = np.array([2.0, 1.5, 0.2, -1.0])
sp = sparsemax(z)
print(f"321 Sparsemax({z.tolist()}) = {sp.round(3).tolist()} (есть нули — РАЗРЕЖЕНО)")
print(f"    Softmax для сравнения = {softmax(z).round(3).tolist()}")

# === 322: Hierarchical softmax (мини-пример: 4 класса = бинарное дерево из 2 уровней) ===
# дерево:           root
#                  /    \
#               n1         n2
#              / \        / \
#             c0 c1      c2 c3
# вероятность класса = произведение sigmoid решений по пути
left_logits = np.array([0.5, -0.3, 1.0])     # узлы: root, n1, n2
def class_prob(c, logits):
    # путь к классу c: root(idx0) -> (n1=0 / n2=1) -> (cN=0 / cN=1)
    go_right_at_root = c >= 2
    go_right_at_n = c % 2 == 1
    p_root = sigmoid(logits[0]) if go_right_at_root else (1 - sigmoid(logits[0]))
    inner = logits[2] if go_right_at_root else logits[1]
    p_inner = sigmoid(inner) if go_right_at_n else (1 - sigmoid(inner))
    return p_root * p_inner

probs = np.array([class_prob(c, left_logits) for c in range(4)])
print(f"322 Hierarchical softmax: P(c)={probs.round(3).tolist()}, sum={probs.sum():.3f}")
print(f"    Сложность O(log V) — для огромных словарей word2vec")
