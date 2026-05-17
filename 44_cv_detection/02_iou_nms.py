"""Раздел 44 — Object Detection.

Файл 2: IoU и NMS.

Концепция: IoU (Intersection over Union).
Площадь пересечения / площадь объединения двух боксов. 0 — нет пересечения,
1 — идентичны. Стандарт для оценки качества детекции и для NMS.

Концепция: NMS (Non-Maximum Suppression).
Из десятков предсказаний на одно лицо/машину оставляем одно. Алгоритм:
1) сортируем по score; 2) берём лучший, добавляем в результат;
3) удаляем все с IoU > threshold; 4) повторяем.

Концепция: Soft-NMS.
Вместо удаления уменьшаем score соседних боксов: s = s * exp(-IoU^2 / sigma).
Меньше false negative при наложенных объектах (толпа).

Концепция: GIoU / DIoU / CIoU.
Расширения IoU, штрафующие за расстояние центров (DIoU) и форму (CIoU).
Лучше как функция потерь, чем чистый IoU.
"""
import numpy as np


def iou(b1, b2):
    """Боксы (x1, y1, x2, y2)."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


b1 = (0, 0, 10, 10)
b2 = (5, 5, 15, 15)
print("IoU двух боксов:", round(iou(b1, b2), 3), "(пересечение 25, объединение 175)")

b3 = (0, 0, 10, 10)
b4 = (20, 20, 30, 30)
print("IoU без пересечения:", iou(b3, b4))

# NMS
boxes = np.array([
    (10, 10, 50, 50),   # score 0.9
    (12, 12, 48, 48),   # score 0.85, перекрывает 0
    (60, 60, 100, 100),  # score 0.8
    (62, 62, 102, 102),  # score 0.75, перекрывает 2
    (15, 15, 55, 55),   # score 0.7, перекрывает 0
], dtype=float)
scores = np.array([0.9, 0.85, 0.8, 0.75, 0.7])


def nms(boxes, scores, iou_thr=0.5):
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        rest = []
        for j in order[1:]:
            if iou(boxes[i], boxes[j]) < iou_thr:
                rest.append(j)
        order = np.array(rest, dtype=int)
    return keep


kept = nms(boxes, scores, 0.5)
print("NMS оставил индексы:", kept, "| scores:", scores[kept])

# Soft-NMS (gaussian)
def soft_nms(boxes, scores, sigma=0.5, score_thr=0.3):
    scores = scores.copy()
    order = list(np.argsort(-scores))
    keep = []
    while order:
        i = order.pop(0)
        if scores[i] < score_thr:
            continue
        keep.append(i)
        for j in order:
            ov = iou(boxes[i], boxes[j])
            scores[j] *= np.exp(-(ov ** 2) / sigma)
        order.sort(key=lambda k: -scores[k])
    return keep, scores


kept_s, new_scores = soft_nms(boxes, scores.copy(), sigma=0.5)
print("Soft-NMS оставил:", kept_s, "| обновлённые scores:", new_scores.round(2))

# GIoU: GIoU = IoU - |C \ (A u B)| / |C|, где C — наименьший охватывающий бокс
def giou(b1, b2):
    cx1 = min(b1[0], b2[0])
    cy1 = min(b1[1], b2[1])
    cx2 = max(b1[2], b2[2])
    cy2 = max(b1[3], b2[3])
    C = (cx2 - cx1) * (cy2 - cy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = a1 + a2 - inter
    return inter / union - (C - union) / C


print("GIoU для b1, b2:", round(giou(b1, b2), 3))
print("GIoU для несоприкасающихся:", round(giou(b3, b4), 3), "(<0)")
