# Vanilla-ML-DL — 1139 концепций ML/DL «на пальцах»

Учебный проект для тех, кто только начинает (и для тех, кто хочет
систематизировать знания). Внутри — **1139 названных концепций**
машинного обучения и Deep Learning (точный счёт по docstring всех файлов,
см. [`CONCEPTS_2.md`](CONCEPTS_2.md)), разобранных короткими самостоятельными
Python-скриптами.

Никаких чёрных ящиков: всё, что можно — реализовано вручную, на чистом
`numpy`, с подробными русскими комментариями. От «что такое вектор» до
PPO, диффузионных моделей, GP и каузального вывода.

## Содержимое одной строкой

- **70 разделов** от базовой математики до причинно-следственного вывода
- **332 запускаемых скрипта**, ~18 700 строк кода
- **1139 концепций** с заголовком и кратким объяснением в docstring
- Только 3 зависимости: `numpy`, `matplotlib`, `scikit-learn`
- Все скрипты smoke-протестированы, каждый отрабатывает за < 5 секунд

## Что нужно

* Python 3.9 или новее
* `pip install -r requirements.txt`

## Как запускать

Каждый файл — самостоятельный. Открой любой и просто запусти:

```bash
python 01_basics/01_vectors.py
python 38_transformers/01_blocks.py
python 70_causal_inference/03_methods.py
```

Меню со списком всех демо:

```bash
python run_all.py
```

## Структура проекта

### Базовый блок (для школьника) — главы 01–11, 119 концепций

1. `01_basics/` — векторы, матрицы, нормы, градиенты, статистика
2. `02_data/` — train/test, нормализация, кодирование, выбросы
3. `03_metrics/` — MAE, F1, ROC-AUC, log-loss, силуэт
4. `04_linear/` — линейная и логистическая регрессия, GD/SGD, softmax
5. `05_regularization/` — L1, L2, Elastic Net, early stopping, dropout
6. `06_classic/` — KNN, NB, деревья, бэггинг, бустинг, SVM, перцептрон
7. `07_unsupervised/` — k-means, иерархическая, DBSCAN, PCA, SVD, t-SNE
8. `08_nn_basics/` — нейрон, активации, backprop вручную, Adam
9. `09_architectures/` — MLP, conv, pooling, RNN, LSTM, GRU, autoencoder
10. `10_optim_training/` — bias/variance, overfitting, learning curves
11. `11_extras/` — Bayes, MLE/MAP, KL, Q-learning, генетика, отжиг

### Продвинутый блок — главы 12–70, ~1900 концепций

| Глава | Тема |
|---|---|
| 12 | Распределения вероятностей (Bernoulli → Dirichlet, MVN) |
| 13 | Статистические тесты (t-test, ANOVA, KS, FDR) |
| 14 | Метрики расстояний (Euclidean → Wasserstein) |
| 15 | Feature engineering (encodings, n-grams, datetime) |
| 16 | Preprocessing (scalers, power transforms, pipelines) |
| 17 | Расширенные линейные модели (Huber, RANSAC, Bayesian, GLM) |
| 18 | Семейство SVM (kernels, OneClass, Nu-SVM) |
| 19 | Расширенные деревья (CART, C4.5, pruning, ExtraTrees) |
| 20 | Бустинг (AdaBoost, GB, XGBoost, LightGBM, CatBoost) |
| 21 | Расширенная кластеризация (K-Medoids, GMM+EM, OPTICS, Fuzzy) |
| 22 | Понижение размерности (LDA, FA, ICA, NMF, ISOMAP, LLE) |
| 23 | Матричная факторизация (SVD, NMF, CF, FM) |
| 24 | Ассоциативные правила (Apriori, support/confidence/lift) |
| 25 | Поиск аномалий (Isolation Forest, LOF, Mahalanobis, AE) |
| 26 | Semi-supervised (self-training, label propagation) |
| 27 | Self-supervised (InfoNCE, SimCLR, MoCo, MAE) |
| 28 | Слои нейросетей (Conv variants, ConvTranspose, pooling) |
| 29 | Расширенные активации (GELU, Swish, Mish, GLU, SwiGLU) |
| 30 | Оптимизаторы (Adam, AdamW, Lion, LAMB, Lookahead, SWA) |
| 31 | Функции потерь (Huber, Focal, Dice, Triplet, InfoNCE) |
| 32 | Расширенная регуляризация (MixUp, CutMix, DropBlock, SAM) |
| 33 | Инициализация (Xavier, He, Orthogonal, Variance Scaling) |
| 34 | Нормализация (Batch, Layer, Instance, Group, RMS) |
| 35 | CNN архитектуры (LeNet → ResNet → EfficientNet → ViT) |
| 36 | RNN (Bi-, Stacked, Beam search, Top-k/Top-p) |
| 37 | Attention (Bahdanau, multi-head, RoPE, ALiBi) |
| 38 | Transformer (encoder/decoder, PreNorm, KV cache, MoE) |
| 39 | Классический NLP (tokenize, BoW, TF-IDF, LDA, HMM) |
| 40 | Word embeddings (word2vec, GloVe, FastText, SIF) |
| 41 | Предобученные LM (BERT, GPT, T5, ELECTRA, RLHF, DPO) |
| 42 | CV основы (image as tensor, im2col, FFT conv) |
| 43 | Классический CV (Canny, Sobel, Harris, HOG, LBP) |
| 44 | Детекция (R-CNN, YOLO, SSD, IoU, NMS, FPN) |
| 45 | Сегментация (FCN, U-Net, SegNet, Dice, Tversky) |
| 46 | Классические генеративные (VAE, GAN, WGAN, CycleGAN) |
| 47 | Современные генеративные (Flows, DDPM, DDIM, CFG) |
| 48 | Расширенный RL (Bellman, SARSA, A2C, PPO, DDPG, SAC) |
| 49 | MLOps (reproducibility, splits, leakage, drift) |
| 50 | Интерпретируемость (PDP/ICE, LIME, SHAP, IG, Grad-CAM) |
| 51 | Graph ML (PageRank, GCN, GraphSAGE, GAT) |
| 52 | Временные ряды (ARIMA, Holt-Winters, ACF/PACF) |
| 53 | Рекомендации (CF, MF, BPR, MAP@K, NDCG) |
| 54 | Байесовский ML (priors, conjugate, Bayesian LR, GP) |
| 55 | MCMC и VI (MH, Gibbs, importance, HMC, mean-field) |
| 56 | Оптимизация (Nesterov, ADMM, FISTA, L-BFGS, CG) |
| 57 | HPO (Bayesian opt, TPE, Hyperband, CMA-ES) |
| 58 | Теория информации (entropy, MI, KL, JS, Fisher) |
| 59 | Линейная алгебра (LU/QR/Cholesky/Schur, Lanczos) |
| 60 | Расширенный матан (Jacobian, Hessian, Taylor, autodiff) |
| 61 | Расширенная теория вероятностей (моменты, Markov chains) |
| 62 | Трюки обучения (mixed precision, EMA, distillation, prune) |
| 63 | Аугментации данных (image/text/audio/tabular, SMOTE) |
| 64 | Meta-learning (transfer, few-shot, MAML, Proto) |
| 65 | Metric learning (Siamese, Triplet, ArcFace) |
| 66 | Active learning (uncertainty, margin, QBC, BatchBALD) |
| 67 | Online learning (PA, FTRL, bandits, UCB, Thompson) |
| 68 | Kernel methods (KRR, kPCA, Nyström, RFF) |
| 69 | Density estimation (KDE, GMM, density ratio) |
| 70 | Causal inference (RCT, IPW, DiD, RDD, uplift) |

## Как устроен каждый файл

Один файл = одна тема. В docstring перечислены все концепции в этом файле
с кратким объяснением (3-6 строк на каждую), дальше идёт демонстрационный
код, который их использует и печатает результат:

```python
"""Раздел N — Тема.

Концепция X: Название.
Короткое объяснение: что это, зачем, как работает, где встречается.

Концепция X+1: ...
"""
import numpy as np

# === Концепция X: ... ===
... демонстрационный код ...
print("X:", результат)
```

Открываешь файл → читаешь docstring → запускаешь → видишь числа →
меняешь параметры → видишь, как меняются числа. Это и есть обучение.

## Что почитать рядом

- [`CONCEPTS.md`](CONCEPTS.md) — список концепций базового блока (1–119)
- [`CONCEPTS_2.md`](CONCEPTS_2.md) — **точная карта**: какой файл содержит
  какие концепции, со ссылками. Автогенерируется скриптом
  `scripts/build_concepts.py` из docstring всех файлов.
- [`STYLE_GUIDE.md`](STYLE_GUIDE.md) — правила оформления

## С чего начать

Не пытайся осилить весь репозиторий за один заход. Выбери одну из дорожек:

### «Я школьник и слышу про ML впервые»
Иди подряд по главам 01 → 02 → 03 → 04 → 06 → 08 → 09.
Этого хватит, чтобы понимать примерно 80% разговоров про ML.

### «Я хочу собрать своими руками нейросеть»
`08_nn_basics/01_neuron.py` → `04_backprop_manual.py` → `12_minibatch_training.py`
→ `09_architectures/01_mlp.py` → `28_neural_layers/` → `30_optimizers_extended/`.

### «Меня позвали на собес по классическому ML»
06 (KNN, NB, деревья, бустинг) + 07 (k-means, PCA) + 19 (деревья глубже)
+ 20 (бустинг подробно) + 50 (интерпретируемость).

### «Хочу разобраться в Transformer и LLM»
37 (attention) → 38 (transformer) → 39 (классический NLP) → 40 (embeddings)
→ 41 (BERT/GPT/T5, RLHF, DPO, inference).

### «Хочу копать в RL»
11/10 (Q-learning) → 48_rl_extended (Bellman → SARSA → A2C → PPO → SAC).

### «Я уже работаю в ML, хочу залатать пробелы»
Открой `CONCEPTS_2.md`, найди тему, которую слышал, но не уверен, что
знаешь. Открой соответствующий файл, прочитай docstring, запусти,
поменяй параметры. Повторяй.

## Совет

Открой любой файл, прочитай docstring, запусти, поменяй параметры,
посмотри, что изменилось. Это и есть обучение.
