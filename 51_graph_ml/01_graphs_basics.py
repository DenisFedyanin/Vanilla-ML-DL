"""Раздел 51 — Graph ML. Файл 1: Основы графов.

Концепция: Матрица смежности A.
A[i,j] = 1, если есть ребро i-j. Симметрична для неориентированных графов.
Удобна для матричных операций, но требует O(n^2) памяти.

Концепция: Степень вершины и матрица степеней D.
deg(i) = sum_j A[i,j]. D = diag(deg) — диагональная матрица степеней.
Сумма степеней = 2 * число рёбер.

Концепция: Лапласиан L = D - A.
Симметричный, положительно полуопределённый. Кратность нулевого собственного
значения = число компонент связности. Используется в спектральной кластеризации.

Концепция: Нормализованный Лапласиан.
L_sym = I - D^(-1/2) A D^(-1/2). Спектр в [0, 2]. Устойчив к разным степеням
вершин. Основа GCN.

Концепция: BFS (поиск в ширину).
Обходит граф уровень за уровнем от стартовой вершины. Очередь FIFO.
Находит кратчайший путь (по числу рёбер) в невзвешенном графе.

Концепция: DFS (поиск в глубину).
Идёт 'вглубь', откатывается по стеку. Используется для топологической сортировки,
поиска компонент связности, циклов.
"""
from collections import deque

import numpy as np

# Маленький граф: 6 вершин
A = np.array([
    [0, 1, 1, 0, 0, 0],
    [1, 0, 1, 1, 0, 0],
    [1, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 0, 1],
    [0, 0, 0, 1, 1, 0],
])
n = A.shape[0]

deg = A.sum(axis=1)
D = np.diag(deg)
print(f"Степени вершин: {deg}")
print(f"Рёбер всего   : {A.sum() // 2}")

L = D - A
eigs = np.sort(np.linalg.eigvalsh(L))
print(f"Eig(L)        : {eigs.round(3)}  (0 один раз => связный граф)")

# Нормализованный Лапласиан
d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1))
L_sym = np.eye(n) - (d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :])
eigs_n = np.sort(np.linalg.eigvalsh(L_sym))
print(f"Eig(L_sym)    : {eigs_n.round(3)}  (в [0, 2])")


def bfs(A, start):
    visited = [False] * len(A)
    order = []
    q = deque([start])
    visited[start] = True
    while q:
        v = q.popleft()
        order.append(v)
        for u in np.where(A[v] == 1)[0]:
            if not visited[u]:
                visited[u] = True
                q.append(u)
    return order


def dfs(A, start):
    visited = [False] * len(A)
    order = []
    stack = [start]
    while stack:
        v = stack.pop()
        if visited[v]:
            continue
        visited[v] = True
        order.append(v)
        for u in reversed(np.where(A[v] == 1)[0]):
            if not visited[u]:
                stack.append(u)
    return order


print(f"BFS из 0: {bfs(A, 0)}")
print(f"DFS из 0: {dfs(A, 0)}")
