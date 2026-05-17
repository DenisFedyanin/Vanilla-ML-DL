"""Раздел 45 — Сегментация.

Файл 3: функции потерь для сегментации.

Концепция: Dice loss.
1 - 2*|A∩B|/(|A|+|B|). Прямо оптимизирует Dice/F1. Помогает при сильном
дисбалансе (объект <5% пикселей), где BCE плохо учит редкий класс.

Концепция: BCE + Dice.
Сумма двух: BCE отвечает за пиксельную точность, Dice — за общую форму.
Часто стабильнее обучается, чем чистый Dice.

Концепция: Tversky loss.
Обобщение Dice: T = TP / (TP + alpha*FP + beta*FN). При alpha<beta штрафует
FN сильнее — хорошо для медицины, где пропуск опухоли хуже false alarm.

Концепция: Lovasz loss.
Суррогат для прямой оптимизации IoU. Строится из субмодулярной функции
ошибок и интегрируется через extension Lovasz. Точная замена IoU.

Концепция: CRF post-processing.
Conditional Random Fields сглаживают грубые предсказания, заставляя
соседние пиксели одного цвета принадлежать одному классу. DenseCRF
популярен с DeepLab v1-v2.
"""
import numpy as np

rng = np.random.default_rng(0)
# GT и предсказание (5x5 бинарная маска)
gt = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
], dtype=float)
pred = np.array([
    [0.1, 0.1, 0.1, 0.1, 0.1],
    [0.1, 0.8, 0.9, 0.7, 0.1],
    [0.1, 0.9, 0.95, 0.85, 0.1],
    [0.1, 0.6, 0.8, 0.3, 0.1],
    [0.1, 0.1, 0.1, 0.1, 0.1],
])


def dice_loss(p, y):
    inter = (p * y).sum()
    return 1 - 2 * inter / (p.sum() + y.sum() + 1e-9)


def bce(p, y):
    return -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))


def tversky(p, y, alpha=0.3, beta=0.7):
    tp = (p * y).sum()
    fp = (p * (1 - y)).sum()
    fn = ((1 - p) * y).sum()
    return 1 - tp / (tp + alpha * fp + beta * fn + 1e-9)


print(f"Dice loss = {dice_loss(pred, gt):.3f}")
print(f"BCE loss  = {bce(pred, gt):.3f}")
print(f"BCE+Dice  = {bce(pred, gt) + dice_loss(pred, gt):.3f}")
print(f"Tversky (a=0.3,b=0.7, штраф FN сильнее) = {tversky(pred, gt):.3f}")
print(f"Tversky (a=0.7,b=0.3, штраф FP сильнее) = {tversky(pred, gt, 0.7, 0.3):.3f}")

# Lovasz hinge — упрощённая реализация для бинарной задачи
def lovasz_hinge(scores, labels):
    """scores — логиты, labels in {0, 1}."""
    signs = 2 * labels - 1
    errors = 1 - scores * signs
    order = np.argsort(-errors)
    errors_sorted = errors[order]
    labels_sorted = labels[order]
    # Lovasz extension
    p = labels_sorted.sum()
    intersection = p - np.cumsum(labels_sorted)
    union = p + np.cumsum(1 - labels_sorted)
    jaccard = 1 - intersection / union
    if len(jaccard) > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return (np.maximum(errors_sorted, 0) * jaccard).sum()


# Логиты от вероятностей
logits = 2 * pred - 1
lv = lovasz_hinge(logits.ravel(), gt.ravel())
print(f"Lovasz hinge loss = {lv:.3f}")

# CRF — концепт: pairwise smoothing
print("\nCRF: unary = -log p_pixel, pairwise = вес*(цвет похож)*(метки разные)")
print("Минимизация энергии E делает соседние пиксели одного цвета одного класса")
