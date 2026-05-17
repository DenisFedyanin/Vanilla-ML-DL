"""Раздел 31 — Лоссы (расширенно). Файл 3: сегментация.

Концепция 351: Dice loss = 1 - 2|A ∩ B| / (|A|+|B|).
Метрика наложения масок. Хорошо работает при сильном дисбалансе фон/объект.

Концепция 352: Tversky loss.
Обобщение Dice: 1 - TP / (TP + alpha*FP + beta*FN).
alpha+beta=1: при beta>alpha штрафуем FN сильнее (полезно для маленьких объектов).

Концепция 353: IoU (Jaccard) loss.
1 - |A ∩ B| / |A ∪ B| = 1 - TP/(TP+FP+FN). Метрика-loss напрямую.

Концепция 354: BCE+Dice combo.
Сумма BCE и Dice — попиксельный градиент + глобальное перекрытие масок.
Стандарт в Kaggle-сегментации.
"""
import numpy as np

rng = np.random.default_rng(0)
H, W = 6, 6
gt = np.zeros((H, W), dtype=float)
gt[2:5, 2:5] = 1
pred_logits = rng.standard_normal((H, W))
pred_logits[2:5, 2:5] += 2.0          # модель в основном правильная
def sigmoid(z): return 1 / (1 + np.exp(-z))
pred = sigmoid(pred_logits)

eps = 1e-7

# === 351: Dice ===
def dice_loss(p, y):
    num = 2 * (p * y).sum()
    den = p.sum() + y.sum() + eps
    return 1 - num / den

print(f"351 Dice loss = {dice_loss(pred, gt):.4f}")

# === 352: Tversky ===
def tversky_loss(p, y, alpha=0.3, beta=0.7):
    tp = (p * y).sum()
    fp = (p * (1 - y)).sum()
    fn = ((1 - p) * y).sum()
    return 1 - tp / (tp + alpha * fp + beta * fn + eps)

print(f"352 Tversky(a=0.3,b=0.7) = {tversky_loss(pred, gt):.4f} (штрафует FN сильнее)")

# === 353: IoU loss ===
def iou_loss(p, y):
    inter = (p * y).sum()
    union = (p + y - p * y).sum()
    return 1 - inter / (union + eps)

print(f"353 IoU loss = {iou_loss(pred, gt):.4f}")

# === 354: BCE + Dice combo ===
def bce(p, y):
    return -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)).mean()

combo = bce(pred, gt) + dice_loss(pred, gt)
print(f"354 BCE+Dice = {combo:.4f} (BCE={bce(pred, gt):.4f} + Dice={dice_loss(pred, gt):.4f})")

# Покажем, как Dice "видит" перекрытие
bin_pred = (pred > 0.5).astype(float)
inter = (bin_pred * gt).sum()
print(f"\nИнтуиция: пикс. наложение {int(inter)}/{int(gt.sum())} целевых, Dice по бинару = {1 - dice_loss(bin_pred, gt):.3f}")
