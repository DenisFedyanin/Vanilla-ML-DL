"""Раздел 45 — Сегментация.

Файл 2: instance и panoptic.

Концепция: Semantic vs Instance vs Panoptic.
Semantic — на каждом пикселе только КЛАСС (две машины слиты в одну маску
'car'). Instance — отдельные id для каждого экземпляра (car_1, car_2).
Panoptic — объединяет: stuff-классы (небо, дорога) семантически, things
(машины, люди) экземплярно.

Концепция: Mask R-CNN для instance.
Faster R-CNN + параллельная mask-голова на каждом RoI. Маска KxK
бинарная для каждого предсказанного класса.

Концепция: Panoptic Segmentation.
Один объединённый выход: для каждого пикселя пара (class_id, instance_id).
Stuff: instance_id = 0. Things: id = 1, 2, ...

Концепция: PQ метрика (Panoptic Quality).
PQ = SQ * RQ, где SQ — средний IoU матчей, RQ = TP / (TP + 0.5*FP + 0.5*FN).
Учитывает и точность матчинга, и отсутствие лишних/пропущенных instance.
"""
import numpy as np

# Маленькая 6x6 сцена с ручной разметкой
sem = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 2, 2],
    [0, 1, 1, 0, 2, 2],
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0, 0],
])  # 0=фон, 1=car, 2=person
print("Semantic карта (классы):")
print(sem)
# Instance id: первая машина (rows 1-2), вторая (rows 4-5); человек один.
inst = np.zeros_like(sem)
inst[1:3, 1:3] = 1   # car_1
inst[4:6, 1:4] = 2   # car_2
inst[1:3, 4:6] = 3   # person_1
print("Instance ids:")
print(inst)

# Panoptic: (class_id, instance_id) per pixel
pan = np.stack([sem, inst], axis=-1)
print("Panoptic пример пикселя (1,1):", tuple(pan[1, 1]),
      "| пикселя (4,1):", tuple(pan[4, 1]))

# Mask R-CNN (концепт): на каждом RoI 28x28 бинарная маска
rng = np.random.default_rng(0)
roi_masks = rng.uniform(size=(3, 28, 28)) > 0.5  # три instance
print("Mask R-CNN: 3 RoI, маски формы:", roi_masks.shape,
      "| доля '1' в маске #0:", roi_masks[0].mean().round(2))

# Простая PQ-метрика между GT и предсказанием на нашей сцене
def iou_mask(a, b):
    inter = (a & b).sum()
    union = (a | b).sum()
    return inter / (union + 1e-9)


# GT instance masks (по id 1, 2, 3 в inst)
gt_masks = [(inst == i) for i in (1, 2, 3)]
# Предсказание: чуть смещённое
pred_masks = []
m = np.zeros_like(sem, dtype=bool)
m[1:3, 1:3] = True
pred_masks.append(m)
m = np.zeros_like(sem, dtype=bool)
m[4:6, 2:4] = True  # сдвинуто
pred_masks.append(m)
# не предсказываем person -> FN

matched = 0
sum_iou = 0
for g in gt_masks:
    best = 0
    for p in pred_masks:
        best = max(best, iou_mask(g, p))
    if best > 0.5:
        matched += 1
        sum_iou += best
tp = matched
fp = len(pred_masks) - matched
fn = len(gt_masks) - matched
SQ = sum_iou / max(tp, 1)
RQ = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-9)
PQ = SQ * RQ
print(f"PQ метрика: TP={tp}, FP={fp}, FN={fn}, SQ={SQ:.2f}, RQ={RQ:.2f}, PQ={PQ:.2f}")
