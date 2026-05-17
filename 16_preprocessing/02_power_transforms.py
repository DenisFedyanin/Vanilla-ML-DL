"""Раздел 16 — Препроцессинг.

Файл 2: power-преобразования и дискретизация.

Концепция 237: QuantileTransformer.
Каждое значение заменяется на ранг/квантиль => равномерное распределение
на [0,1]; либо дальше пропускаем через inverse CDF нормали — получим
почти-нормальное. Очень робастен к выбросам.

Концепция 238: PowerTransformer (Yeo-Johnson).
Подбирает lambda, минимизирующую асимметрию (макс. правдоподобие нормальности).
Работает с любыми числами (в отличие от Box-Cox только для >0).

Концепция 239: KBinsDiscretizer.
Дискретизация в k бинов тремя способами: 'uniform' (равная ширина),
'quantile' (равная частота), 'kmeans' (центры — кластеры одномерного k-means).
"""
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer, PowerTransformer, QuantileTransformer

rng = np.random.default_rng(0)
X = rng.lognormal(mean=0.0, sigma=1.0, size=(200, 1))  # длинный правый хвост

# 237: QT to uniform
qt_u = QuantileTransformer(n_quantiles=50, output_distribution="uniform", random_state=0).fit_transform(X)
print(f"237a QT->uniform: min={qt_u.min():.3f}, max={qt_u.max():.3f}, mean={qt_u.mean():.3f}")

# 237: QT to normal
qt_n = QuantileTransformer(n_quantiles=50, output_distribution="normal", random_state=0).fit_transform(X)
print(f"237b QT->normal:  mean={qt_n.mean():.3f}, std={qt_n.std():.3f}")

# 238: PowerTransformer (yeo-johnson)
pt = PowerTransformer(method="yeo-johnson").fit(X)
X_pt = pt.transform(X)
print(f"238 PowerTransformer lam={pt.lambdas_[0]:.3f}, после: mean={X_pt.mean():.3f}, skew~0")

# 239: KBinsDiscretizer
for strat in ("uniform", "quantile", "kmeans"):
    kb = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy=strat).fit(X)
    out = kb.transform(X[:5]).ravel().astype(int)
    print(f"239 KBins({strat}): первые 5 бинов = {out.tolist()}")
