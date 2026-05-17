"""Раздел 58 — Теория информации.

Файл 3: Продвинутые понятия.

Концепция: Информация Фишера.
I(theta) = E[(d ln p/d theta)^2]. Для N(mu, sigma^2 фикс) по mu: I = 1/sigma^2.
Граница Крамера-Рао: Var(оценка mu) >= 1/(n*I) = sigma^2/n (достигается выборочным средним).

Концепция: Связь Cross-Entropy <-> KL.
H(p,q) = -sum p log q = H(p) + KL(p||q). Минимизация cross-entropy
эквивалентна минимизации KL(true || pred), т.к. H(p) от модели не зависит.

Концепция: Information Bottleneck (IB).
Принцип: найти представление T от X, максимизирующее I(T;Y) - beta*I(T;X).
T должно сохранять информацию о Y, но "забывать" лишнее в X. Применяется в DL для интерпретации.

Концепция: MIC (Maximal Information Coefficient).
Нормированная мера зависимости. Бьет данные на сетку m*n, считает MI по эмпирической p(x,y),
нормирует на log min(m,n). Максимизирует по сеткам с m*n<=B(N). Улавливает разные виды зависимостей.

Концепция: MINE (Mutual Information Neural Estimation).
Оценка I(X;Y) через нейросеть T: I_MINE = sup_T E_{p_xy}[T] - log E_{p_x p_y}[exp T].
Это нижняя граница из Donsker-Varadhan. На практике T обучают градиентным спуском.
"""
import numpy as np

rng = np.random.default_rng(0)

# === Fisher для N(mu, sigma^2=1) ===
n = 500
mu_true, sigma = 1.5, 1.0
x = rng.normal(mu_true, sigma, n)
mu_hat = x.mean()
fisher = 1 / sigma ** 2
crb = 1 / (n * fisher)
emp_var = (x - mu_hat).var(ddof=1) / n
print(f"Fisher info (mu)= {fisher}, CRB={crb:.4f}, эмп. var(mu_hat)={emp_var:.4f}")

# === CE = H(p) + KL(p||q) ===
p = np.array([0.2, 0.5, 0.3])
q = np.array([0.1, 0.6, 0.3])
H_p = -np.sum(p * np.log(p))
CE = -np.sum(p * np.log(q))
KL = np.sum(p * np.log(p / q))
print(f"CE={CE:.4f}, H(p)+KL(p||q)={H_p+KL:.4f}")

# === IB (концептуальная иллюстрация) ===
# Сожмем X -> T так, чтобы T предсказывало Y. На двоичной задаче:
# X in {0..7}, Y = X%2. Лучшее T = (X mod 2) сохраняет I(T;Y)=1 бит, теряет всё лишнее.
X = np.arange(8)
Y = X % 2
T = X % 2
def H_disc(v):
    _, c = np.unique(v, return_counts=True); p = c / c.sum()
    return -(p * np.log2(p)).sum()
def MI(a, b):
    return H_disc(a) + H_disc(b) - H_disc(list(zip(a, b)))
print(f"IB: I(T;Y)={MI(T,Y):.3f}, I(T;X)={MI(T,X):.3f} (T сжимает X, сохраняя Y)")

# === MIC: грубый расчет на сетке ===
n = 200
x = rng.uniform(0, 1, n)
y = np.sin(6 * x) + 0.05 * rng.normal(size=n)
def mi_grid(x, y, m=4, k=4):
    bx = np.linspace(x.min(), x.max() + 1e-9, m + 1)
    by = np.linspace(y.min(), y.max() + 1e-9, k + 1)
    ix = np.digitize(x, bx) - 1
    iy = np.digitize(y, by) - 1
    joint = np.zeros((m, k))
    for a, b in zip(ix, iy): joint[a, b] += 1
    joint /= joint.sum()
    px = joint.sum(1, keepdims=True); py = joint.sum(0, keepdims=True)
    mask = joint > 0
    return (joint[mask] * np.log2(joint[mask] / (px @ py)[mask])).sum()
mics = [mi_grid(x, y, m, k) / np.log2(min(m, k)) for m, k in [(2,2),(3,3),(4,4),(5,5)]]
print(f"MIC ~ max по сеткам = {max(mics):.3f}")

# === MINE (концепт): покажем вид оценки на готовой T(x,y)=x*y ===
xy = np.column_stack([x, y])
shuffle = y[rng.permutation(n)]
T_joint = (x * y)
T_marg = (x * shuffle)
mine_lb = T_joint.mean() - np.log(np.exp(T_marg).mean())
print(f"MINE-нижняя оценка I(X;Y) с T(x,y)=xy: {mine_lb:.3f} (натов)")
