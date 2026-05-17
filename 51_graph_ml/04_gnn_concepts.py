"""Раздел 51 — Graph ML. Файл 4: GNN — концепции.

Концепция: Message Passing Framework.
GNN-слой: для каждой вершины собираем сообщения от соседей m_uv = MSG(h_u, h_v),
агрегируем (sum/mean/max) -> a_v, обновляем: h_v <- UPDATE(h_v, a_v). Все
архитектуры (GCN, GraphSAGE, GAT) — варианты этой схемы.

Концепция: GCN (Graph Convolutional Network).
Слой: H' = sigma( hat_A H W ),  hat_A = D^(-1/2) (A + I) D^(-1/2).
Self-loop (+I) гарантирует, что вершина сохраняет свою информацию.
Нормировка по корням степеней — устойчивость к разным deg.

Концепция: GraphSAGE.
Вместо использования всех соседей сэмплируем фиксированное число k_l на каждом
слое. Агрегатор может быть mean, max, LSTM. Позволяет обучаться индуктивно
(на новых, не виденных вершинах), масштабируется на крупные графы.

Концепция: GAT (Graph Attention Network).
Внимание вместо нормировки: alpha_uv = softmax_u( LeakyReLU(a^T [W h_u || W h_v]) ).
h_v <- sigma( sum_u alpha_uv W h_u ). Разные соседи получают разные веса —
'важные' влияют сильнее.

Концепция: Многослойные GNN.
k слоёв = k-hop рецептивное поле. Глубокие GNN страдают oversmoothing:
эмбеддинги всех вершин сходятся к одинаковому. Решения: skip connections,
ограничение глубины, normalization.
"""
import numpy as np

rng = np.random.default_rng(0)

# Маленький граф 5 вершин, 3 фичи
A = np.array([
    [0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0],
    [1, 1, 0, 0, 0],
    [0, 1, 0, 0, 1],
    [0, 0, 0, 1, 0],
], dtype=float)
n = 5
H = rng.normal(size=(n, 3))

# === GCN forward ===
A_hat = A + np.eye(n)
d = A_hat.sum(axis=1)
D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
W1 = rng.normal(scale=0.5, size=(3, 4))
H1 = np.maximum(0, A_norm @ H @ W1)
print(f"GCN слой: H -> H1 shape {H1.shape}")
print(f"H1[0] (v0 эмбеддинг): {H1[0].round(2)}")

# === GraphSAGE (mean-aggregator с сэмплированием) ===
def sage_layer(H, A, W_self, W_neigh, k=2):
    out = np.zeros((len(H), W_self.shape[1]))
    for v in range(len(H)):
        nbrs = np.where(A[v] == 1)[0]
        if len(nbrs) > k:
            nbrs = rng.choice(nbrs, size=k, replace=False)
        agg = H[nbrs].mean(axis=0) if len(nbrs) else np.zeros_like(H[0])
        out[v] = np.maximum(0, H[v] @ W_self + agg @ W_neigh)
    return out


Ws = rng.normal(scale=0.3, size=(3, 4))
Wn = rng.normal(scale=0.3, size=(3, 4))
H_sage = sage_layer(H, A, Ws, Wn, k=2)
print(f"GraphSAGE H[0] = {H_sage[0].round(2)}  (сэмплируем 2 соседа)")

# === GAT (одна голова) ===
W = rng.normal(scale=0.3, size=(3, 4))
a = rng.normal(scale=0.3, size=8)        # [Wh_u || Wh_v] -> скаляр
WH = H @ W
H_gat = np.zeros_like(WH)
for v in range(n):
    nbrs = np.append(np.where(A[v] == 1)[0], v)   # +self
    scores = []
    for u in nbrs:
        cat = np.concatenate([WH[v], WH[u]])
        scores.append(np.maximum(0.2 * (a @ cat), a @ cat))   # LeakyReLU(0.2)
    scores = np.array(scores)
    alpha = np.exp(scores - scores.max())
    alpha = alpha / alpha.sum()
    H_gat[v] = sum(alpha[i] * WH[u] for i, u in enumerate(nbrs))
print(f"GAT H[0] = {H_gat[0].round(2)}  (взвешенная агрегация)")

# Oversmoothing: применим GCN 10 раз — эмбеддинги становятся похожими
H_deep = H.copy()
W_id = np.eye(3)
for _ in range(10):
    H_deep = A_norm @ H_deep @ W_id
spread = np.std(H_deep, axis=0).mean()
print(f"После 10 GCN слоёв std эмбеддингов = {spread:.4f}  (близко к 0 — oversmoothing)")
