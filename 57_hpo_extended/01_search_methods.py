"""Раздел 57 — HPO (расширение).

Файл 1: Базовые методы поиска.

Концепция: Grid Search.
Перебор по декартовому произведению значений каждого гиперпараметра.
Гарантированно покрывает сетку, но число точек экспоненциально растет от размерности.

Концепция: Random Search.
Случайный выбор гиперпараметров из распределений. При том же бюджете часто
находит лучшие точки, особенно когда лишь 1-2 параметра реально важны.

Концепция: Почему random выигрывает при низкой effective dimensionality.
Если из k параметров реально влияет только один, то grid n^k точек дает
лишь n значений по важной оси. Random N точек дает N разных значений по любой оси.
"""
import numpy as np

rng = np.random.default_rng(0)

# Целевая функция: важна только первая координата, остальные шум
def objective(p):
    return -(p[0] - 0.3) ** 2 + 0.01 * rng.normal()

# === Grid search 5x5 по 2D ===
grid = [(a, b) for a in np.linspace(0, 1, 5) for b in np.linspace(0, 1, 5)]
best_g = max(grid, key=lambda p: -((p[0] - 0.3) ** 2))
print(f"Grid 5x5: лучшая точка p0={best_g[0]:.3f} (true=0.3), значений по p0={len({p[0] for p in grid})}")

# === Random search 25 точек ===
rand = [(rng.uniform(), rng.uniform()) for _ in range(25)]
best_r = max(rand, key=lambda p: -((p[0] - 0.3) ** 2))
print(f"Random 25: лучшая точка p0={best_r[0]:.3f}, уникальных p0={len({round(p[0],4) for p in rand})}")

# === Демонстрация: при k=5, влияет только p0 ===
def obj5(p):
    return -((p[0] - 0.3) ** 2)

# Grid 3 на ось -> 3^5=243 точки, но всего 3 значения по p0
N = 243
grid5_vals_p0 = 3
random5_pts = rng.uniform(size=(N, 5))
random5_best = obj5(random5_pts[np.argmax([obj5(p) for p in random5_pts])])
grid_best = -((np.linspace(0, 1, 3) - 0.3) ** 2).min()
print(f"5D, бюджет {N}: grid лучшее ={grid_best:.4f}, random лучшее ={random5_best:.4f}")
print("Random видит ~243 разных значений по p0, grid — только 3.")
