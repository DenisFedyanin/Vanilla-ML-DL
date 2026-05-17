"""Раздел 44 — Object Detection.

Файл 3: семейство R-CNN.

Концепция: R-CNN.
1) Selective Search генерирует ~2000 region proposals. 2) Каждый кроп
ресайзится, прогоняется через CNN. 3) SVM классифицирует. 4) Регрессор
уточняет бокс. Медленно (CNN на каждый кроп).

Концепция: Fast R-CNN.
CNN считается ОДИН РАЗ для всего изображения. Region proposals проектируются
на feature map. RoI Pooling — фиксированный размер фичей для произвольного бокса.
Multi-task loss: classification + bbox regression.

Концепция: Faster R-CNN.
Selective Search заменяется RPN (Region Proposal Network) — маленькой свёрткой
поверх feature map. Anchors разных масштабов и aspect ratio предсказывают
objectness + offset. Полностью end-to-end и быстрее.

Концепция: Mask R-CNN.
Faster R-CNN + параллельная "маска"-голова (FCN на RoI) для instance segmentation.
RoI Align (без квантования координат) даёт точные пиксельные маски.

Концепция: RoI Pooling vs RoI Align.
Pooling: квантует координаты RoI до целых -> сдвиг ~0.5 пикселя. Align:
билинейная интерполяция, без квантования. Критично для масок.
"""
import numpy as np

rng = np.random.default_rng(0)

# R-CNN: пайплайн (концепт)
n_proposals = 8
proposals = rng.integers(0, 200, size=(n_proposals, 4))
proposals[:, 2:] = proposals[:, :2] + rng.integers(20, 80, size=(n_proposals, 2))
print("R-CNN: число region proposals:", n_proposals,
      "| каждый ресайзится в 224x224 и прогоняется через CNN")

# Fast R-CNN: feature map один раз
feat_map_shape = (1, 256, 14, 14)  # B, C, H, W
print("Fast R-CNN feature map форма:", feat_map_shape,
      "-> RoI pooling до 7x7 для каждого proposal")


# RoI Pooling (упрощённо)
def roi_pool(feat, roi, out_size=2):
    """feat (C,H,W); roi (x1,y1,x2,y2) в координатах feat."""
    C, H, W = feat.shape
    x1, y1, x2, y2 = roi
    out = np.zeros((C, out_size, out_size))
    bin_h = (y2 - y1) / out_size
    bin_w = (x2 - x1) / out_size
    for i in range(out_size):
        for j in range(out_size):
            sy1 = int(np.floor(y1 + i * bin_h))
            sy2 = int(np.ceil(y1 + (i + 1) * bin_h))
            sx1 = int(np.floor(x1 + j * bin_w))
            sx2 = int(np.ceil(x1 + (j + 1) * bin_w))
            sy2, sx2 = max(sy2, sy1 + 1), max(sx2, sx1 + 1)
            patch = feat[:, sy1:sy2, sx1:sx2]
            if patch.size:
                out[:, i, j] = patch.max(axis=(1, 2))
    return out


feat = rng.normal(size=(3, 10, 10))
roi_p = roi_pool(feat, (1.5, 1.5, 6.7, 6.7), out_size=2)
print("RoI Pooling вывод форма:", roi_p.shape, "(C=3, 2x2)")


# RoI Align: билинейная интерполяция
def bilinear_sample(feat, y, x):
    C, H, W = feat.shape
    y0, x0 = int(np.floor(y)), int(np.floor(x))
    y1_, x1_ = y0 + 1, x0 + 1
    y0, x0 = max(0, min(H - 1, y0)), max(0, min(W - 1, x0))
    y1_, x1_ = max(0, min(H - 1, y1_)), max(0, min(W - 1, x1_))
    fy, fx = y - np.floor(y), x - np.floor(x)
    v = ((1 - fy) * (1 - fx) * feat[:, y0, x0] + (1 - fy) * fx * feat[:, y0, x1_]
         + fy * (1 - fx) * feat[:, y1_, x0] + fy * fx * feat[:, y1_, x1_])
    return v


def roi_align(feat, roi, out_size=2):
    x1, y1, x2, y2 = roi
    out = np.zeros((feat.shape[0], out_size, out_size))
    bin_h = (y2 - y1) / out_size
    bin_w = (x2 - x1) / out_size
    for i in range(out_size):
        for j in range(out_size):
            cy = y1 + (i + 0.5) * bin_h
            cx = x1 + (j + 0.5) * bin_w
            out[:, i, j] = bilinear_sample(feat, cy, cx)
    return out


roi_a = roi_align(feat, (1.5, 1.5, 6.7, 6.7), out_size=2)
print("RoI Align вывод форма:", roi_a.shape,
      "| RoI Pooling vs Align max|diff|:", np.abs(roi_p - roi_a).max().round(3))

# RPN: anchors
anchor_scales = [16, 32, 64]
anchor_ratios = [0.5, 1.0, 2.0]
n_anchors = len(anchor_scales) * len(anchor_ratios)
print("RPN: на каждой точке feature map", n_anchors,
      "anchors (3 масштаба x 3 соотношения)")
print("Faster R-CNN end-to-end: feature -> RPN proposals -> RoIAlign -> head (cls+bbox)")
print("Mask R-CNN: дополнительно FCN-голова -> бинарная маска KxK на RoI")
