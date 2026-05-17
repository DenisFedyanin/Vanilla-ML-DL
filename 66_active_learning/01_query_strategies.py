"""Раздел 66 — Active Learning.

Файл 1: Стратегии query (отбор точек для разметки).

Концепция: Active Learning.
У нас много неразмеченных данных и ограниченный бюджет на разметку.
Активное обучение: модель сама выбирает, какие объекты разметить, чтобы быстрее
улучшить качество. Цикл: train -> query -> label -> add to train -> repeat.

Концепция: Uncertainty sampling (least confidence).
Берём объект, у которого МАКСИМАЛЬНАЯ вероятность класса минимальна:
x* = argmin_x max_c P(y=c | x). Модель в нём наименее уверена.

Концепция: Margin sampling.
Берём объект, у которого мала разница между топ-1 и топ-2 классами:
x* = argmin_x [P(top1|x) - P(top2|x)]. Точки на границе решений.

Концепция: Entropy sampling.
Берём объект с максимальной энтропией предсказания:
x* = argmax_x H(P(y|x)) = -sum P log P. Самая 'размазанная' уверенность по классам.

Демо: на 2D-задаче с логистической регрессией сравниваем кривые обучения
для случайного отбора vs uncertainty sampling.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

rng = np.random.default_rng(0)

# === игрушечный пул ===
X, y = make_classification(n_samples=400, n_features=4, n_informative=3,
                           n_redundant=0, n_classes=3, n_clusters_per_class=1,
                           random_state=0)
X_test, y_test = make_classification(n_samples=300, n_features=4, n_informative=3,
                                     n_redundant=0, n_classes=3, n_clusters_per_class=1,
                                     random_state=1)

def init_split():
    idx = rng.permutation(len(X))
    labeled = list(idx[:10])
    pool = list(idx[10:])
    return labeled, pool

# === Стратегии ===
def least_confidence(probs):
    return np.argmin(probs.max(1))

def margin(probs):
    sorted_p = np.sort(probs, axis=1)[:, ::-1]
    return np.argmin(sorted_p[:, 0] - sorted_p[:, 1])

def entropy(probs):
    eps = 1e-12
    H = -(probs * np.log(probs + eps)).sum(1)
    return np.argmax(H)

def run_active(strategy, n_iter=20):
    labeled, pool = init_split()
    accs = []
    for _ in range(n_iter):
        model = LogisticRegression(max_iter=500).fit(X[labeled], y[labeled])
        accs.append(model.score(X_test, y_test))
        if strategy == "random":
            j = rng.integers(0, len(pool))
        else:
            probs = model.predict_proba(X[pool])
            if strategy == "least_conf":
                j = least_confidence(probs)
            elif strategy == "margin":
                j = margin(probs)
            elif strategy == "entropy":
                j = entropy(probs)
        labeled.append(pool.pop(j))
    return np.array(accs)

acc_rand = run_active("random")
acc_lc = run_active("least_conf")
acc_marg = run_active("margin")
acc_ent = run_active("entropy")

print("Active Learning curve (acc после k запросов):")
print(f"  random      : start={acc_rand[0]:.3f}, mid={acc_rand[10]:.3f}, end={acc_rand[-1]:.3f}")
print(f"  least_conf  : start={acc_lc[0]:.3f}, mid={acc_lc[10]:.3f}, end={acc_lc[-1]:.3f}")
print(f"  margin      : start={acc_marg[0]:.3f}, mid={acc_marg[10]:.3f}, end={acc_marg[-1]:.3f}")
print(f"  entropy     : start={acc_ent[0]:.3f}, mid={acc_ent[10]:.3f}, end={acc_ent[-1]:.3f}")
print(f"Прирост uncertainty vs random на финале: "
      f"{acc_lc[-1] - acc_rand[-1]:+.3f}")
