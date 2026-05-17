"""Раздел 41 — Предобученные LM. Файл 3: alignment (SFT, RLHF, DPO).

Концепция: Instruction tuning (SFT).
Дообучаем LM на парах (промпт, желаемый ответ). Перевод из предсказателя текста
в "следующего инструкциям ассистента". Маленький набор качественных пар.

Концепция: Reward model.
Регрессионная модель, обученная на парах (a > b): предсказывает скаляр.
Источник данных — human preferences. Обучается через ranking loss
(Bradley-Terry: log σ(r(a) - r(b))).

Концепция: RLHF (PPO).
Используем reward model как сигнал; policy = LM с KL-штрафом к SFT-модели,
чтобы не "сходить с ума". Алгоритм — PPO.

Концепция: DPO (Direct Preference Optimization).
Эквивалентная reward model + RL формулировка, но без RL:
loss = -log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x))).
Прямо учим политику предпочитать y_w (winner) над y_l (loser).
"""
import numpy as np
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(0)

# === SFT ===
print("=== SFT (instruction tuning) ===")
sft_data = [
    ("переведи: cat", "кот"),
    ("столица Франции?", "Париж"),
    ("2+2", "4"),
]
print("Обучаем next-token loss только на ответе:")
for prompt, ans in sft_data:
    print(f"  prompt='{prompt}' -> target='{ans}' (loss считается только по target)")

# === Reward model на hand-crafted preference pairs ===
print("\n=== Reward model ===")
# Скажем, "хорошие" ответы имеют признаки [полезность, точность, длина_оптим]
# Препочтения: a > b означает, что фичи a в среднем "лучше" чем у b
n_pairs = 30
# Генерируем фичи победителей и проигравших
winners = rng.standard_normal((n_pairs, 3)) + np.array([1.0, 1.0, 0.0])
losers = rng.standard_normal((n_pairs, 3)) + np.array([-0.5, 0.0, 0.0])
# Обучаем регрессию: винеры — целевая 1, лузеры — 0
X = np.vstack([winners, losers])
y = np.array([1] * n_pairs + [0] * n_pairs).astype(float)
rm = LinearRegression().fit(X, y)
print(f"Веса reward model: {np.round(rm.coef_, 2)}")
print(f"reward winner mean = {rm.predict(winners).mean():.2f}; loser = {rm.predict(losers).mean():.2f}")

# === RLHF (PPO) — концепция ===
print("\n=== RLHF (PPO) ===")
print("policy = LM_θ; reference = LM_SFT (заморожен)")
print("на каждом шаге: семплируем y ~ π_θ(·|x), считаем reward r = RM(x,y)")
print("обновляем θ через PPO с KL-штрафом: maximize E[r] - β KL(π_θ || π_ref)")

# === DPO loss demonstration ===
print("\n=== DPO ===")
# Имитируем log-вероятности из reference и policy для пары (y_w, y_l)
log_p_ref_w = -2.0
log_p_ref_l = -2.0
log_p_pol_w = -1.5    # policy предпочитает y_w
log_p_pol_l = -2.5
beta = 1.0
margin = beta * ((log_p_pol_w - log_p_ref_w) - (log_p_pol_l - log_p_ref_l))
dpo_loss = -np.log(1 / (1 + np.exp(-margin)))
print(f"margin = β * (Δlog π_pol_w - Δlog π_pol_l) = {margin:.2f}")
print(f"DPO loss = -log σ(margin) = {dpo_loss:.3f}")
print("Если margin > 0 — модель предпочитает y_w, как мы и хотим.")
