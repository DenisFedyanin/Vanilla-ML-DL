"""Раздел 14 — Метрики расстояний.

Файл 2: расстояния для МНОЖЕСТВ, СТРОК, последовательностей и геокоординат.

Концепция 186: Jaccard.
1 - |A ∩ B| / |A ∪ B|. Расстояние между множествами. 0 если совпадают,
1 если не пересекаются. Применимо к keyword sets, shingles.

Концепция 187: Dice (Sorensen-Dice).
1 - 2|A ∩ B| / (|A| + |B|). Похоже на Jaccard, но "мягче": пересечение
учитывается с весом 2. Часто используется в сегментации (Dice loss).

Концепция 188: Hamming.
Число позиций, в которых строки/векторы одинаковой длины различаются.
Применимо к бинарным кодам, хешам, ДНК.

Концепция 189: Levenshtein (edit distance).
Минимальное число вставок/удалений/замен символов для превращения одной
строки в другую. Реализуется через DP-таблицу O(|s||t|).

Концепция 190: Dynamic Time Warping (DTW).
"Эластичное" расстояние между двумя временными рядами разной длины.
Ищем оптимальное выравнивание через DP. Незаменимо в распознавании речи/жестов.

Концепция 191: Haversine.
Расстояние по дуге большого круга на сфере (для широт/долгот в радианах):
a = sin^2(dlat/2) + cos(lat1)*cos(lat2)*sin^2(dlon/2). d = 2R*asin(sqrt(a)).
"""
import numpy as np

A = {"кот", "собака", "птица"}
B = {"собака", "птица", "рыба", "ёж"}

# 186: Jaccard
jac_sim = len(A & B) / len(A | B)
print(f"186 Jaccard dist: {1 - jac_sim:.4f}")

# 187: Dice
dice_sim = 2 * len(A & B) / (len(A) + len(B))
print(f"187 Dice dist: {1 - dice_sim:.4f}")

# 188: Hamming
s1 = "karolin"
s2 = "kathrin"
ham = sum(a != b for a, b in zip(s1, s2))
print(f"188 Hamming({s1},{s2})={ham}")

# 189: Levenshtein DP
def levenshtein(a, b):
    n, m = len(a), len(b)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return int(dp[n, m])


print(f"189 Levenshtein('kitten','sitting')={levenshtein('kitten','sitting')}")

# 190: DTW DP
def dtw(a, b):
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[n, m]


x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 4.0])
y = np.array([1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0])
print(f"190 DTW={dtw(x, y):.4f} (L1 после warp)")

# 191: Haversine, Москва ↔ Питер
def haversine(lat1, lon1, lat2, lon2, R=6371.0):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


print(f"191 Haversine MSK-SPB={haversine(55.75, 37.62, 59.94, 30.31):.1f} км")
