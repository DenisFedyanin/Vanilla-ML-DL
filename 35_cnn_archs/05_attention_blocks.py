"""Раздел 35 — CNN. Файл 5: внимание в CNN, ConvNeXt, ViT.

Концепция: Squeeze-and-Excitation (SE) block.
Берём feature map (B,C,H,W) -> GlobalAvgPool по H,W -> (B,C) ->
FC down -> ReLU -> FC up -> sigmoid -> (B,C) -> умножаем на исходные каналы.
Сеть учит "важность каналов" — внимание по канальной оси.

Концепция: CBAM (Convolutional Block Attention Module).
Channel attention (как SE, но и max-pool, и avg-pool) +
Spatial attention (avg/max по каналам -> 7x7 conv -> sigmoid -> карта внимания H,W).
Последовательно применяется к feature map.

Концепция: ConvNeXt.
"Современная" свёрточная сеть, вдохновлённая ViT:
большие kernel (7x7) depthwise, LayerNorm, GELU, инвертированный bottleneck.
Догоняет ViT при том же бюджете.

Концепция: Vision Transformer (ViT).
Разбиваем изображение на патчи P×P, разворачиваем каждый в вектор,
линейно проектируем, добавляем positional embedding,
прогоняем через стек Transformer encoder.
[CLS]-токен или mean-pool используется для классификации.
"""
import numpy as np

rng = np.random.default_rng(0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# === SE block: manual ===
print("=== Squeeze-and-Excitation ===")
B, C, H, W = 1, 16, 8, 8
x = rng.standard_normal((B, C, H, W))
# 1) Squeeze: GAP
s = x.mean(axis=(2, 3))                                      # (B, C)
# 2) Excitation: FC -> ReLU -> FC -> sigmoid
r = 4
W1 = rng.standard_normal((C, C // r)) * 0.1
W2 = rng.standard_normal((C // r, C)) * 0.1
h = np.maximum(0, s @ W1)
gate = sigmoid(h @ W2)                                       # (B, C) в (0,1)
# 3) Scale: умножаем каждый канал
y = x * gate[:, :, None, None]
print(f"x{x.shape} -> GAP{s.shape} -> gate{gate.shape} -> y{y.shape}")
print(f"средние веса каналов: min={gate.min():.3f}, max={gate.max():.3f}")

# === CBAM (channel + spatial attention) ===
print("\n=== CBAM ===")
# Channel attention: max-pool и avg-pool по H,W, MLP, сумма, sigmoid
avg = x.mean(axis=(2, 3))
mx = x.max(axis=(2, 3))
ch_att = sigmoid(avg @ W1 @ W2 + mx @ W1 @ W2)
x1 = x * ch_att[:, :, None, None]
# Spatial attention: avg и max по каналам -> (B,2,H,W) -> 7x7 conv (имитируем суммой) -> sigmoid
sp_avg = x1.mean(axis=1, keepdims=True)
sp_max = x1.max(axis=1, keepdims=True)
sp_att = sigmoid(sp_avg + sp_max)              # (B,1,H,W)
y_cbam = x1 * sp_att
print(f"channel_att{ch_att.shape}, spatial_att{sp_att.shape}, out{y_cbam.shape}")

# === ConvNeXt блок (формы) ===
print("\n=== ConvNeXt block (форма) ===")
C = 96
print(f"depthwise 7x7 (C={C}) -> LayerNorm -> 1x1 (C->4C) -> GELU -> 1x1 (4C->C) -> + residual")

# === ViT patches ===
print("\n=== Vision Transformer patchify ===")
img = rng.standard_normal((1, 3, 32, 32))
P = 8
Bv, Cv, Hv, Wv = img.shape
n_patches = (Hv // P) * (Wv // P)
# reshape image to patches
patches = img.reshape(Bv, Cv, Hv // P, P, Wv // P, P).transpose(0, 2, 4, 1, 3, 5)
patches = patches.reshape(Bv, n_patches, Cv * P * P)
print(f"image{img.shape} -> {n_patches} patches размера {P}x{P}, dim={Cv*P*P}")
print(f"patches{patches.shape}")
# линейная проекция в embed_dim
embed_dim = 64
Wp = rng.standard_normal((Cv * P * P, embed_dim)) * 0.02
tokens = patches @ Wp
# CLS token + positional embedding
cls = rng.standard_normal((Bv, 1, embed_dim)) * 0.02
seq = np.concatenate([cls, tokens], axis=1)
pos = rng.standard_normal((1, seq.shape[1], embed_dim)) * 0.02
seq = seq + pos
print(f"tokens{tokens.shape} + CLS -> seq{seq.shape}")
