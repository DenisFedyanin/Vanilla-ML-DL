"""Раздел 58 — Теория информации.

Файл 2: Дивергенции между распределениями.

Концепция: KL-дивергенция.
KL(P||Q) = sum p log(p/q). Не симметрична, не метрика.
Бесконечна если q(x)=0 при p(x)>0. Связь с MLE и cross-entropy.

Концепция: Дивергенция Йенсена-Шеннона (JS).
JS(P,Q) = 0.5 KL(P||M) + 0.5 KL(Q||M), M=(P+Q)/2. Симметрична, всегда конечна,
sqrt(JS) — метрика. Часто используется как мягкая альтернатива KL.

Концепция: Хеллингер.
H(P,Q) = (1/sqrt(2)) ||sqrt(p)-sqrt(q)||_2 ∈ [0,1]. Симметрична, метрика.

Концепция: Total Variation (TV).
TV(P,Q) = 0.5 sum |p-q|. Самая "интуитивная" — максимальная разность вероятностей события.
Связь с других: TV<=sqrt(KL/2) (Пинскер), TV<=sqrt(2)*H.

Концепция: Бхаттачарья.
BC(P,Q) = sum sqrt(p*q), B-дистанция = -ln(BC). Связана с Хеллингером:
H^2 = 1 - BC.
"""
import numpy as np

p = np.array([0.1, 0.4, 0.5])
q = np.array([0.2, 0.5, 0.3])

eps = 1e-12

def KL(p, q):
    return np.sum(p * np.log((p + eps) / (q + eps)))

def JS(p, q):
    m = 0.5 * (p + q)
    return 0.5 * KL(p, m) + 0.5 * KL(q, m)

hellinger = (1 / np.sqrt(2)) * np.linalg.norm(np.sqrt(p) - np.sqrt(q))
tv = 0.5 * np.abs(p - q).sum()
bc = np.sum(np.sqrt(p * q))
bdist = -np.log(bc)

print(f"KL(P||Q) = {KL(p, q):.4f}")
print(f"KL(Q||P) = {KL(q, p):.4f}   (несимметрично)")
print(f"JS(P,Q)  = {JS(p, q):.4f}")
print(f"Hellinger= {hellinger:.4f}")
print(f"TV       = {tv:.4f}")
print(f"BC       = {bc:.4f},  B-дист = {bdist:.4f}")
print(f"Проверка H^2 = 1-BC: {hellinger**2:.4f} vs {1-bc:.4f}")

# Pinsker: TV <= sqrt(KL/2)
print(f"Pinsker: TV={tv:.4f} <= sqrt(KL/2)={np.sqrt(KL(p,q)/2):.4f}")

# Когда p=q -> все дивергенции 0
print(f"При p=q все дивергенции 0: KL={KL(p,p):.2e}, JS={JS(p,p):.2e}")
