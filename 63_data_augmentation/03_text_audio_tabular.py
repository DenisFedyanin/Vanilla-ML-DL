"""Раздел 63 — Data augmentation.

Файл 3: Text, audio, tabular.

Концепция: Text — synonym replacement.
Случайно заменяем слова на синонимы из словаря. Простейший EDA-приём.

Концепция: Text — random deletion / swap.
Случайно удаляем слова с вероятностью p или меняем местами случайные пары соседних.

Концепция: Text — back-translation.
Переводим текст на язык X и обратно. Получается перефразированный вариант.
Требует двух моделей переводчиков, но дает разнообразные перефразы.

Концепция: Audio — time stretch.
Изменение длительности без изменения высоты тона (через phase vocoder). Тренирует
инвариантность к темпу.

Концепция: Audio — pitch shift.
Изменение высоты тона без изменения длительности. Используется в музыке/речи.

Концепция: Audio — SpecAugment.
На спектрограмме маскируем случайные полосы по времени (T-mask) и по частотам (F-mask).
Стало стандартом в ASR.

Концепция: Tabular — SMOTE.
Для миноритарного класса: для каждого x берем k ближайших соседей того же класса,
случайно выбираем одного x_n и сэмплируем x_new = x + alpha*(x_n - x), alpha~U(0,1).

Концепция: Tabular — ADASYN.
Похож на SMOTE, но больше синтетики генерирует возле "трудных" примеров —
тех, у кого среди соседей много мажоритарного класса.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors

rng = np.random.default_rng(0)

# === Text augs ===
syn = {"быстро": ["скоро", "оперативно"], "хороший": ["неплохой", "приличный"]}
words = "это хороший пример быстро написанного текста".split()

# synonym replacement
out = words.copy()
for i, w in enumerate(out):
    if w in syn and rng.uniform() < 0.5:
        out[i] = rng.choice(syn[w])
print(f"SynonymRepl: {' '.join(out)}")

# random deletion p=0.2
out = [w for w in words if rng.uniform() > 0.2 or len(words) <= 1]
print(f"RandomDel:   {' '.join(out)}")

# random swap (одна пара)
out = words.copy()
i = rng.integers(len(out) - 1)
out[i], out[i + 1] = out[i + 1], out[i]
print(f"RandomSwap:  {' '.join(out)}")
print("BackTrans (концепт): RU -> EN -> RU дает перефраз")

# === Audio (спектрограмма как 2D) — SpecAugment ===
spec = rng.uniform(0, 1, (12, 30))  # 12 mel x 30 time
# T-mask: занулим 5 идущих временных кадров
t0 = rng.integers(0, 30 - 5)
spec_aug = spec.copy(); spec_aug[:, t0:t0 + 5] = 0
# F-mask: занулим 3 частотных полосы
f0 = rng.integers(0, 12 - 3)
spec_aug[f0:f0 + 3, :] = 0
print(f"SpecAugment: маски T=[{t0}:{t0+5}], F=[{f0}:{f0+3}], нули = {(spec_aug==0).sum()} ячеек")
print("Time-stretch/pitch-shift (концепт): через STFT и сдвиг фаз/частот.")

# === Tabular SMOTE ===
# Имитация бинарного дисбаланса
X_maj = rng.normal(0, 1, size=(50, 2))
X_min = rng.normal(2, 0.5, size=(8, 2))
nn = NearestNeighbors(n_neighbors=4).fit(X_min)

new_pts = []
for x in X_min:
    _, idxs = nn.kneighbors(x.reshape(1, -1))
    n = X_min[rng.choice(idxs[0][1:])]  # один из соседей (не сам)
    alpha = rng.uniform()
    new_pts.append(x + alpha * (n - x))
X_smote = np.vstack([X_min, np.array(new_pts)])
print(f"SMOTE: minority {len(X_min)} -> {len(X_smote)} (синтетических {len(new_pts)})")

# === ADASYN-концепт: больше синтетики возле "трудных" ===
nn_all = NearestNeighbors(n_neighbors=5).fit(np.vstack([X_maj, X_min]))
hardness = []
for x in X_min:
    _, idxs = nn_all.kneighbors(x.reshape(1, -1))
    # доля мажоритарных среди соседей
    maj_count = sum(1 for j in idxs[0] if j < len(X_maj))
    hardness.append(maj_count / 5.0)
hardness = np.array(hardness); hardness /= hardness.sum() + 1e-9
print(f"ADASYN: hardness-распределение для синтеза = {hardness.round(2)}")
