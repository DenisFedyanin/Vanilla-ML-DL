"""Раздел 31 — Лоссы (расширенно). Файл 2: классификационные.

Концепция 347: BCE (Binary Cross-Entropy).
L = -[y*log p + (1-y)*log(1-p)]. p из sigmoid(logit).
Стандарт бинарной классификации. Минимум при p=y.

Концепция 348: CE (Categorical Cross-Entropy).
L = -sum_k y_k * log p_k, где y — one-hot. Эквивалент NLL после softmax.
Базовый лосс мульти-классов.

Концепция 349: Focal loss.
L = -alpha * (1-p_t)^gamma * log p_t, p_t = вер-ть правильного класса.
Снижает вес "лёгких" примеров; полезно при дисбалансе (детекция).

Концепция 350: Label-smoothing CE.
y_smooth = (1-eps) * y_onehot + eps / K. Не даёт логитам уезжать в бесконечность,
делает модель калиброваннее.
"""
import numpy as np

def sigmoid(z): return 1 / (1 + np.exp(-z))
def softmax(z):
    s = z - z.max(axis=-1, keepdims=True)
    e = np.exp(s); return e / e.sum(axis=-1, keepdims=True)

# === 347: BCE ===
y = np.array([1, 0, 1, 1, 0])
logits = np.array([2.0, -1.0, 0.5, 3.0, 1.0])
p = sigmoid(logits)
eps = 1e-9
bce = -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)).mean()
print(f"347 BCE = {bce:.4f}; p={p.round(3).tolist()}")

# === 348: CE ===
K = 3
logits_c = np.array([[2.0, 0.5, -1.0],
                     [-1.0, 3.0, 0.2],
                     [0.1, 0.2, 0.0]])
labels = np.array([0, 1, 2])
P = softmax(logits_c)
ce = -np.log(P[np.arange(len(labels)), labels] + eps).mean()
print(f"348 CE  = {ce:.4f}")

# === 349: Focal loss (gamma=2, alpha=0.25) ===
def focal(P, labels, gamma=2.0, alpha=0.25):
    pt = P[np.arange(len(labels)), labels]
    return -(alpha * (1 - pt) ** gamma * np.log(pt + eps)).mean()

print(f"349 Focal(gamma=2)= {focal(P, labels):.4f}")
print(f"    Сравните: лёгкий пример pt=0.95 даёт CE=-log 0.95 = {-np.log(0.95):.3f},")
print(f"    Focal: {(1 - 0.95) ** 2 * -np.log(0.95):.6f} — почти 0 (он 'лёгкий')")

# === 350: Label smoothing CE (eps=0.1) ===
def smooth_ce(P, labels, K, eps_s=0.1):
    onehot = np.eye(K)[labels]
    q = (1 - eps_s) * onehot + eps_s / K
    return -(q * np.log(P + eps)).sum(axis=1).mean()

ls = smooth_ce(P, labels, K, eps_s=0.1)
print(f"350 LabelSmooth CE(eps=0.1) = {ls:.4f} (>= обычной CE на ту же модель)")
