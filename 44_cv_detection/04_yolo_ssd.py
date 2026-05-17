"""Раздел 44 — Object Detection.

Файл 4: YOLO, SSD, FPN, focal loss.

Концепция: YOLO (You Only Look Once).
Делим изображение на SxS грид. Каждая ячейка предсказывает B боксов
(x, y, w, h, conf) + вероятности классов. Один прогон сети — все детекции.
Очень быстро (real-time).

Концепция: Anchor boxes.
Заранее заданные шаблоны размеров. Сеть предсказывает offsets от anchor,
а не абсолютные координаты. Лучше сходимость и работа с разными формами.

Концепция: SSD (Single Shot Detector).
Одно-проходный детектор с предсказаниями на нескольких уровнях feature map:
мелкие фичи для маленьких объектов, грубые — для крупных.

Концепция: FPN (Feature Pyramid Network).
Top-down путь объединяет богатые семантически верхние фичи с детальными
нижними через upsample + lateral connections. Стандарт для multi-scale.

Концепция: Focal Loss.
FL = -(1 - p_t)^gamma * log(p_t). Усиливает вес сложных примеров, гасит
вклад "лёгких" фоновых. Помогает при дисбалансе класс/фон в RetinaNet.
"""
import numpy as np

rng = np.random.default_rng(0)

# YOLO grid 7x7, B=2 боксов, C=20 классов
S, B, C = 7, 2, 20
yolo_out = rng.normal(size=(S, S, B * 5 + C))
print("YOLO output форма:", yolo_out.shape,
      "| на ячейку:", B * 5 + C, "значений (B*(x,y,w,h,conf) + C)")

# Привязка детекций к ячейкам: целевая ячейка центра объекта
img_h = img_w = 448
obj_cx, obj_cy = 200, 300  # пикселей
cell_size = img_w / S
ci, cj = int(obj_cy // cell_size), int(obj_cx // cell_size)
print(f"Объект в ({obj_cx},{obj_cy}) -> ячейка ({ci}, {cj}), относительные:"
      f" x={obj_cx % cell_size / cell_size:.2f}, y={obj_cy % cell_size / cell_size:.2f}")

# Anchor matching: выбираем anchor с лучшим IoU к GT-боксу по форме
anchors = np.array([[30, 30], [60, 60], [30, 90], [90, 30]])  # (w, h)
gt_wh = np.array([50, 40])
ious = []
for a in anchors:
    inter = min(a[0], gt_wh[0]) * min(a[1], gt_wh[1])
    union = a[0] * a[1] + gt_wh[0] * gt_wh[1] - inter
    ious.append(inter / union)
best_a = int(np.argmax(ious))
print("Anchor IoU:", np.round(ious, 3), "| лучший anchor:", best_a)

# SSD: фичи на разных уровнях
ssd_levels = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
total_boxes = sum(h * w * 6 for h, w in ssd_levels)  # 6 anchors/cell примерно
print("SSD: размеры feature maps:", ssd_levels, "| всего ~", total_boxes, "боксов")

# FPN: топ-даун
def upsample2x(x):
    h, w = x.shape
    return np.repeat(np.repeat(x, 2, axis=0), 2, axis=1)


C5 = rng.normal(size=(4, 4))
C4 = rng.normal(size=(8, 8))
P5 = C5.copy()
P4 = upsample2x(P5) + C4  # lateral + top-down
print("FPN: P5", P5.shape, "-> upsample +", C4.shape, "= P4", P4.shape)


# Focal loss
def focal_loss(p, y, gamma=2.0):
    """p — вероятность положительного, y in {0,1}."""
    p_t = np.where(y == 1, p, 1 - p)
    return -(1 - p_t) ** gamma * np.log(p_t + 1e-9)


probs = np.array([0.9, 0.1, 0.7, 0.99])
labels = np.array([1, 1, 0, 0])  # 1-й — лёгкий, 3-й — очень лёгкий
ce = -np.where(labels == 1, np.log(probs + 1e-9), np.log(1 - probs + 1e-9))
fl = focal_loss(probs, labels, gamma=2.0)
print("CE:", ce.round(3))
print("Focal (gamma=2):", fl.round(3),
      "| лёгкие примеры (0.99 negative, 0.9 positive) почти обнулились")
