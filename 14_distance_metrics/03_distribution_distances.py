"""Раздел 14 — Метрики расстояний.

Файл 3: расстояния между РАСПРЕДЕЛЕНИЯМИ.

Концепция 192: KL-дивергенция.
KL(P||Q) = sum p_i log(p_i/q_i). Несимметрична, "стоимость кодирования P
кодом для Q". 0 когда P=Q. Появляется в VI, t-SNE, кросс-энтропии.

Концепция 193: Jensen-Shannon (JS).
0.5*KL(P||M) + 0.5*KL(Q||M), где M=(P+Q)/2. Симметрична и ограничена.
sqrt(JS) — настоящая метрика.

Концепция 194: Hellinger.
H(P,Q) = sqrt(0.5 * sum (sqrt(p_i)-sqrt(q_i))^2). Настоящая метрика,
тесно связана с Bhattacharyya.

Концепция 195: Bhattacharyya.
BC = sum sqrt(p_i*q_i); distance = -ln(BC). Используется в физике частиц,
теории информации.

Концепция 196: Wasserstein-1D (Earth Mover's).
Для 1D: W1 = sum |F_P(x) - F_Q(x)| dx ~ интеграл разности CDF. "Сколько
работы перевезти массу P, чтобы получить Q". Используется в WGAN.

Концепция 197: Mahalanobis (как расстояние между точкой и распределением).
d_M = sqrt((x-mu)^T Sigma^-1 (x-mu)). Учитывает форму облака точек.
"""
import numpy as np

# 192: KL
P = np.array([0.1, 0.4, 0.5])
Q = np.array([0.2, 0.3, 0.5])
kl_pq = (P * np.log(P / Q)).sum()
kl_qp = (Q * np.log(Q / P)).sum()
print(f"192 KL(P||Q)={kl_pq:.4f}, KL(Q||P)={kl_qp:.4f} (несимметрично)")

# 193: JS
M = 0.5 * (P + Q)
js = 0.5 * (P * np.log(P / M)).sum() + 0.5 * (Q * np.log(Q / M)).sum()
print(f"193 JS(P,Q)={js:.4f}, sqrt(JS)={np.sqrt(js):.4f}")

# 194: Hellinger
H = np.sqrt(0.5 * ((np.sqrt(P) - np.sqrt(Q)) ** 2).sum())
print(f"194 Hellinger={H:.4f}")

# 195: Bhattacharyya
BC = np.sqrt(P * Q).sum()
bd = -np.log(BC)
print(f"195 Bhattacharyya BC={BC:.4f}, dist={bd:.4f}")

# 196: Wasserstein-1D через CDF
rng = np.random.default_rng(0)
a = rng.normal(0, 1, size=500)
b = rng.normal(1, 1, size=500)  # ожидаем W1 ≈ 1
sorted_a = np.sort(a)
sorted_b = np.sort(b)
W1 = np.mean(np.abs(sorted_a - sorted_b))
print(f"196 Wasserstein-1D N(0,1) vs N(1,1) ≈ {W1:.4f} (теор=1)")

# 197: Mahalanobis между точкой и распределением
mu = np.array([0.0, 0.0])
Sigma = np.array([[2.0, 0.8], [0.8, 1.0]])
x = np.array([1.5, 1.0])
diff = x - mu
m = float(np.sqrt(diff @ np.linalg.inv(Sigma) @ diff))
print(f"197 Mahalanobis(x, N(mu, Sigma))={m:.4f}")
