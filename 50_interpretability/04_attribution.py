"""Раздел 50 — Interpretability. Файл 4: Attribution для нейронных сетей.

Концепция: Saliency map.
Градиент выхода по входу: |df/dx_i|. Большие значения — пиксели/фичи, на которые
сеть 'смотрела'. Самый простой attribution. Шумный, чувствителен к насыщению.

Концепция: Integrated Gradients (IG).
IG_i = (x_i - x'_i) * ∫_{α=0..1} ∂f(x' + α(x - x'))/∂x_i dα.
Усредняет градиенты вдоль прямого пути от baseline x' к x. Удовлетворяет аксиомам
полноты: sum(IG) = f(x) - f(x'). Менее шумно, чем чистый saliency.

Концепция: Baseline.
Точка x' с 'нейтральным' предсказанием: чёрная картинка, нулевой вектор, среднее.
Качество IG зависит от выбора baseline.

Концепция: Grad-CAM (для CNN).
Берём активации последнего conv-слоя A_k и градиенты ∂y_c/∂A_k. Усредняем градиенты
по пространству (alpha_k = mean grad). Карта = ReLU(sum_k alpha_k * A_k). Грубее
по разрешению, но устойчивее: показывает 'где' на картинке.

Концепция: Аксиомы attribution.
Sensitivity (если фича различает x и x' и f(x)!=f(x'), у неё ненулевой вклад),
Implementation invariance (одинаковые функции — одинаковое объяснение). IG удовлетворяет
обоим, plain saliency — нет.
"""
import numpy as np

rng = np.random.default_rng(0)

# Игрушечная NN: y = sigmoid(W2 @ tanh(W1 @ x + b1) + b2)
W1 = rng.normal(scale=0.5, size=(4, 3))
b1 = rng.normal(scale=0.1, size=4)
W2 = rng.normal(scale=0.5, size=(1, 4))
b2 = rng.normal(scale=0.1, size=1)


def forward(x):
    h = np.tanh(W1 @ x + b1)
    z = (W2 @ h + b2)[0]
    return 1 / (1 + np.exp(-z))


def grad(x):
    h_pre = W1 @ x + b1
    h = np.tanh(h_pre)
    z = (W2 @ h + b2)[0]
    s = 1 / (1 + np.exp(-z))
    dy_dh = W2[0] * s * (1 - s)         # df/dh
    dh_dx = (1 - h ** 2)[:, None] * W1   # dh/dx
    return dy_dh @ dh_dx


x = np.array([1.0, -0.5, 0.3])
baseline = np.zeros(3)

# Saliency
sal = np.abs(grad(x))
print(f"f(x)={forward(x):.4f}  f(baseline)={forward(baseline):.4f}")
print(f"Saliency |df/dx|              : {sal.round(3)}")

# Integrated Gradients (Riemann-сумма)
steps = 50
ig = np.zeros(3)
for k in range(steps):
    alpha = (k + 0.5) / steps
    ig += grad(baseline + alpha * (x - baseline))
ig = (x - baseline) * ig / steps
print(f"Integrated Gradients          : {ig.round(3)}")
print(f"Полнота: sum(IG)={ig.sum():.4f}  vs f(x)-f(baseline)={forward(x) - forward(baseline):.4f}")

# Grad-CAM концепт: представим h (скрытый слой) как 'feature maps'
h_pre = W1 @ x + b1
h = np.tanh(h_pre)
dy_dh = W2[0] * forward(x) * (1 - forward(x))
alphas = dy_dh                                 # 'average pooled' градиенты (тут 1D)
cam = np.maximum(0, alphas * h)
print(f"Grad-CAM (по скрытым нейронам): {cam.round(3)}  (важные каналы — крупные)")
