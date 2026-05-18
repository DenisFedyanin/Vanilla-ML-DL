# CONCEPTS — карта концепций по файлам

Автогенерируется из docstring каждого файла глав.
Всего 332 файлов, 1139 концепций в 70 разделах.

Регенерация: `python scripts/build_concepts.py`.

## Оглавление

- [01_basics — Математика и numpy](#01-basics) (12 концепций)
- [02_data — Подготовка данных](#02-data) (12 концепций)
- [03_metrics — Метрики качества](#03-metrics) (11 концепций)
- [04_linear — Линейные модели](#04-linear) (8 концепций)
- [05_regularization — Регуляризация](#05-regularization) (5 концепций)
- [06_classic — Классические алгоритмы](#06-classic) (15 концепций)
- [07_unsupervised — Обучение без учителя](#07-unsupervised) (8 концепций)
- [08_nn_basics — Основы нейросетей](#08-nn-basics) (11 концепций)
- [09_architectures — Архитектуры нейросетей](#09-architectures) (10 концепций)
- [10_optim_training — Обучение и оптимизация](#10-optim-training) (10 концепций)
- [11_extras — Дополнительные концепции](#11-extras) (12 концепций)
- [12_probability_distributions — Распределения вероятностей](#12-probability-distributions) (33 концепций)
- [13_statistics_tests — Статистические тесты](#13-statistics-tests) (24 концепций)
- [14_distance_metrics — Метрики расстояний](#14-distance-metrics) (21 концепций)
- [15_feature_engineering — Feature engineering](#15-feature-engineering) (32 концепций)
- [16_preprocessing — Препроцессинг](#16-preprocessing) (17 концепций)
- [17_linear_models_extended — Расширенные линейные модели](#17-linear-models-extended) (17 концепций)
- [18_svm_family — Семейство SVM](#18-svm-family) (10 концепций)
- [19_trees_extended — Расширенные деревья](#19-trees-extended) (12 концепций)
- [20_boosting_family — Бустинг](#20-boosting-family) (21 концепций)
- [21_clustering_extended — Расширенная кластеризация](#21-clustering-extended) (19 концепций)
- [22_dim_reduction_extended — Понижение размерности](#22-dim-reduction-extended) (15 концепций)
- [23_matrix_factorization — Матричная факторизация](#23-matrix-factorization) (14 концепций)
- [24_associations — Ассоциативные правила](#24-associations) (10 концепций)
- [25_anomaly_detection — Поиск аномалий](#25-anomaly-detection) (12 концепций)
- [26_semi_supervised — Semi-supervised обучение](#26-semi-supervised) (12 концепций)
- [27_self_supervised — Self-supervised обучение](#27-self-supervised) (12 концепций)
- [28_neural_layers — Слои нейросетей](#28-neural-layers) (21 концепций)
- [29_activations_extended — Функции активации](#29-activations-extended) (22 концепций)
- [30_optimizers_extended — Оптимизаторы](#30-optimizers-extended) (19 концепций)
- [31_losses_extended — Функции потерь](#31-losses-extended) (17 концепций)
- [32_regularization_extended — Расширенная регуляризация](#32-regularization-extended) (17 концепций)
- [33_init_extended — Инициализация весов](#33-init-extended) (14 концепций)
- [34_normalization — Нормализация](#34-normalization) (10 концепций)
- [35_cnn_archs — Архитектуры CNN](#35-cnn-archs) (21 концепций)
- [36_rnn_extended — Расширенные RNN](#36-rnn-extended) (14 концепций)
- [37_attention — Механизмы attention](#37-attention) (20 концепций)
- [38_transformers — Архитектура Transformer](#38-transformers) (19 концепций)
- [39_nlp_classic — Классический NLP](#39-nlp-classic) (20 концепций)
- [40_nlp_embeddings — Word embeddings](#40-nlp-embeddings) (8 концепций)
- [41_lm_pretrained — Предобученные языковые модели](#41-lm-pretrained) (19 концепций)
- [42_cv_basics — Основы computer vision](#42-cv-basics) (15 концепций)
- [43_cv_classic — Классический computer vision](#43-cv-classic) (19 концепций)
- [44_cv_detection — Детекция объектов](#44-cv-detection) (17 концепций)
- [45_cv_segmentation — Сегментация изображений](#45-cv-segmentation) (14 концепций)
- [46_generative_classic — Классические генеративные модели](#46-generative-classic) (19 концепций)
- [47_generative_modern — Современные генеративные модели](#47-generative-modern) (20 концепций)
- [48_rl_extended — Reinforcement learning](#48-rl-extended) (26 концепций)
- [49_mlops_basics — Основы MLOps](#49-mlops-basics) (22 концепций)
- [50_interpretability — Интерпретируемость моделей](#50-interpretability) (20 концепций)
- [51_graph_ml — Graph ML](#51-graph-ml) (21 концепций)
- [52_time_series — Временные ряды](#52-time-series) (25 концепций)
- [53_recommender — Рекомендательные системы](#53-recommender) (21 концепций)
- [54_bayesian_ml — Байесовский ML](#54-bayesian-ml) (20 концепций)
- [55_mcmc — MCMC и вариационный вывод](#55-mcmc) (15 концепций)
- [56_optimization_extended — Расширенная оптимизация](#56-optimization-extended) (23 концепций)
- [57_hpo_extended — Подбор гиперпараметров](#57-hpo-extended) (15 концепций)
- [58_info_theory — Теория информации](#58-info-theory) (14 концепций)
- [59_linalg_extended — Расширенная линейная алгебра](#59-linalg-extended) (14 концепций)
- [60_calculus_extended — Расширенный матанализ](#60-calculus-extended) (12 концепций)
- [61_probability_extended — Расширенная теория вероятностей](#61-probability-extended) (20 концепций)
- [62_tricks_training — Трюки обучения](#62-tricks-training) (22 концепций)
- [63_data_augmentation — Аугментация данных](#63-data-augmentation) (18 концепций)
- [64_meta_learning — Meta-learning](#64-meta-learning) (9 концепций)
- [65_metric_learning — Metric learning](#65-metric-learning) (10 концепций)
- [66_active_learning — Active learning](#66-active-learning) (9 концепций)
- [67_online_learning — Online learning](#67-online-learning) (15 концепций)
- [68_kernel_methods — Kernel methods](#68-kernel-methods) (15 концепций)
- [69_density_estimation — Density estimation](#69-density-estimation) (14 концепций)
- [70_causal_inference — Causal inference](#70-causal-inference) (19 концепций)

## 01_basics — Математика и numpy

- `01_basics/01_vectors.py`
  - **1.** Векторы, матрицы, тензоры
- `01_basics/02_dot_product.py`
  - **2.** Скалярное произведение (dot product)
- `01_basics/03_norms.py`
  - **3.** Нормы L1, L2, L_inf
- `01_basics/04_cosine_similarity.py`
  - **4.** Косинусное сходство
- `01_basics/05_numerical_gradient.py`
  - **5.** Численная производная и градиент
- `01_basics/06_chain_rule.py`
  - **6.** Цепное правило
- `01_basics/07_monte_carlo_pi.py`
  - **7.** Закон больших чисел через оценку числа Пи методом Монте-Карло
- `01_basics/08_distributions.py`
  - **8.** Распределения - равномерное и нормальное
- `01_basics/09_descriptive_stats.py`
  - **9.** Описательная статистика
- `01_basics/10_correlation.py`
  - **10.** Корреляция Пирсона
- `01_basics/11_probability_rules.py`
  - **11.** Сложение и умножение вероятностей
- `01_basics/12_clt.py`
  - **12.** Центральная предельная теорема (ЦПТ)

## 02_data — Подготовка данных

- `02_data/01_train_test_split.py`
  - **13.** Train / Test split
- `02_data/02_kfold.py`
  - **14.** K-Fold кросс-валидация
- `02_data/03_stratified_split.py`
  - **15.** Стратифицированный split
- `02_data/04_zscore_standardization.py`
  - **16.** Z-score стандартизация
- `02_data/05_minmax.py`
  - **17.** Min-Max нормализация
- `02_data/06_robust_scaling.py`
  - **18.** Robust scaling (по медиане и IQR)
- `02_data/07_one_hot.py`
  - **19.** One-Hot encoding
- `02_data/08_label_encoding.py`
  - **20.** Label encoding
- `02_data/09_missing_values.py`
  - **21.** Заполнение пропусков
- `02_data/10_outliers.py`
  - **22.** Обнаружение выбросов
- `02_data/11_oversampling.py`
  - **23.** Дисбаланс классов и oversampling
- `02_data/12_polynomial_features.py`
  - **24.** Полиномиальные признаки

## 03_metrics — Метрики качества

- `03_metrics/01_mae.py`
  - **25.** MAE - Mean Absolute Error
- `03_metrics/02_mse_rmse.py`
  - **26.** MSE и RMSE
- `03_metrics/03_r2.py`
  - **27.** Коэффициент детерминации R^2
- `03_metrics/04_accuracy.py`
  - **28.** Accuracy
- `03_metrics/05_precision_recall.py`
  - **29.** Precision и Recall
- `03_metrics/06_f1.py`
  - **30.** F1-score
- `03_metrics/07_confusion_matrix.py`
  - **31.** Матрица ошибок (confusion matrix)
- `03_metrics/08_roc_auc.py`
  - **32.** ROC-кривая и AUC
- `03_metrics/09_logloss.py`
  - **33.** Log-loss (бинарная cross-entropy)
- `03_metrics/10_silhouette.py`
  - **34.** Silhouette score (метрика качества кластеризации)
- `03_metrics/11_mape.py`
  - **35.** MAPE - Mean Absolute Percentage Error

## 04_linear — Линейные модели

- `04_linear/01_linear_regression_normal_eq.py`
  - **36.** Линейная регрессия через нормальное уравнение
- `04_linear/02_full_gd.py`
  - **37.** Полный (batch) градиентный спуск
- `04_linear/03_sgd.py`
  - **38.** Стохастический градиентный спуск (SGD)
- `04_linear/04_minibatch_gd.py`
  - **39.** Mini-batch GD
- `04_linear/05_sigmoid.py`
  - **41.** Сигмоида
- `04_linear/06_logistic_regression.py`
  - **40.** Логистическая регрессия (бинарная классификация)
- `04_linear/07_softmax.py`
  - **42.** Softmax
- `04_linear/08_cross_entropy.py`
  - **43.** Многоклассовая cross-entropy

## 05_regularization — Регуляризация

- `05_regularization/01_l2_ridge.py`
  - **45.** L2 регуляризация (Ridge)
- `05_regularization/02_l1_lasso.py`
  - **44.** L1 регуляризация (Lasso)
- `05_regularization/03_elastic_net.py`
  - **46.** Elastic Net = L1 + L2
- `05_regularization/04_early_stopping.py`
  - **47.** Early stopping
- `05_regularization/05_dropout_concept.py`
  - **48.** Dropout - концепция

## 06_classic — Классические алгоритмы

- `06_classic/01_knn.py`
  - **49.** K-Nearest Neighbors
- `06_classic/02_naive_bayes.py`
  - **50.** Gaussian Naive Bayes
- `06_classic/03_entropy_gini.py`
  - **52.** Энтропия и Gini impurity
- `06_classic/04_information_gain.py`
  - **53.** Information Gain
- `06_classic/05_decision_tree.py`
  - **51.** Decision Tree (с нуля, для понимания)
- `06_classic/06_bagging.py`
  - **54.** Bagging (Bootstrap Aggregating)
- `06_classic/07_random_forest.py`
  - **55.** Random Forest
- `06_classic/08_adaboost.py`
  - **56.** AdaBoost - очень упрощённая реализация (decision stumps)
- `06_classic/09_gradient_boosting.py`
  - **57.** Gradient Boosting (для регрессии, идея)
- `06_classic/10_svm_hinge.py`
  - **58.** Линейный SVM через hinge loss
- `06_classic/11_kernel_trick.py`
  - **59.** Kernel trick
- `06_classic/12_perceptron.py`
  - **60.** Перцептрон Розенблатта (1957)
- `06_classic/13_one_vs_rest.py`
  - **61.** One-vs-Rest для мультикласса
- `06_classic/14_voting.py`
  - **62.** Voting classifier
- `06_classic/15_stacking.py`
  - **63.** Stacking (концепт)

## 07_unsupervised — Обучение без учителя

- `07_unsupervised/01_kmeans.py`
  - **64.** K-Means
- `07_unsupervised/02_kmeans_pp.py`
  - **65.** K-Means++ инициализация
- `07_unsupervised/03_hierarchical.py`
  - **66.** Иерархическая (агломеративная) кластеризация
- `07_unsupervised/04_dbscan.py`
  - **67.** DBSCAN
- `07_unsupervised/05_pca.py`
  - **68.** PCA - метод главных компонент
- `07_unsupervised/06_svd.py`
  - **69.** SVD - сингулярное разложение
- `07_unsupervised/07_tsne.py`
  - **70.** t-SNE - метод визуализации многомерных данных в 2D
- `07_unsupervised/08_elbow.py`
  - **71.** Elbow method для выбора K в K-Means

## 08_nn_basics — Основы нейросетей

- `08_nn_basics/01_neuron.py`
  - **72.** Один нейрон и линейный слой
- `08_nn_basics/02_activations.py`  *(нет извлекаемых docstring-концепций)*
- `08_nn_basics/03_forward_pass.py`
  - **78.** Forward pass (прямой проход)
- `08_nn_basics/04_backprop_manual.py`
  - **79.** Обратное распространение ошибки (backprop), руками
- `08_nn_basics/05_init_xavier.py`
  - **80.** Xavier (Glorot) инициализация
- `08_nn_basics/06_init_he.py`
  - **81.** He инициализация
- `08_nn_basics/07_batchnorm.py`
  - **82.** Batch Normalization
- `08_nn_basics/08_dropout_impl.py`
  - **83.** Dropout (реализация в forward/eval)
- `08_nn_basics/09_momentum.py`
  - **85.** SGD с momentum
- `08_nn_basics/10_adam.py`
  - **86.** Adam
- `08_nn_basics/11_lr_schedule.py`
  - **87.** Learning rate schedule
- `08_nn_basics/12_minibatch_training.py`
  - **84.** Mini-batch обучение нейросети

## 09_architectures — Архитектуры нейросетей

- `09_architectures/01_mlp.py`
  - **88.** MLP - многослойный перцептрон
- `09_architectures/02_conv1d.py`
  - **89.** 1D-свёртка - вручную
- `09_architectures/03_conv2d.py`
  - **90.** 2D-свёртка - вручную
- `09_architectures/04_pooling.py`
  - **91.** Max и Average Pooling
- `09_architectures/05_cnn_demo.py`
  - **92.** CNN - концепт (свёртка + ReLU + pooling)
- `09_architectures/06_rnn.py`
  - **93.** RNN-ячейка - вручную
- `09_architectures/07_lstm.py`
  - **94.** LSTM ячейка (концептуально)
- `09_architectures/08_gru.py`
  - **95.** GRU - упрощённый LSTM с 2 гейтами (reset, update)
- `09_architectures/09_autoencoder.py`
  - **96.** Autoencoder
- `09_architectures/10_word_embeddings.py`
  - **97.** Word embeddings

## 10_optim_training — Обучение и оптимизация

- `10_optim_training/01_loss_landscape.py`
  - **98.** Loss landscape
- `10_optim_training/02_vanishing_grad.py`
  - **99.** Vanishing gradient
- `10_optim_training/03_exploding_grad.py`
  - **100.** Exploding gradient
- `10_optim_training/04_gradient_clipping.py`
  - **101.** Gradient clipping
- `10_optim_training/05_bias_variance.py`
  - **102.** Bias-Variance tradeoff
- `10_optim_training/06_under_overfit.py`
  - **103.** Underfitting vs Overfitting
- `10_optim_training/07_learning_curves.py`
  - **104.** Learning curves
- `10_optim_training/08_validation_curves.py`
  - **105.** Validation curves
- `10_optim_training/09_grid_search.py`
  - **106.** Grid Search
- `10_optim_training/10_random_search.py`
  - **107.** Random Search

## 11_extras — Дополнительные концепции

- `11_extras/01_curse_of_dim.py`
  - **108.** Проклятие размерности
- `11_extras/02_feature_importance.py`
  - **109.** Feature importance (через Random Forest)
- `11_extras/03_permutation_importance.py`
  - **110.** Permutation Importance
- `11_extras/04_probability_calibration.py`
  - **111.** Калибровка вероятностей
- `11_extras/05_kl_divergence.py`
  - **112.** KL-дивергенция и связь с cross-entropy
- `11_extras/06_mle.py`
  - **113.** MLE - метод максимального правдоподобия
- `11_extras/07_map.py`
  - **114.** MAP - Maximum A Posteriori
- `11_extras/08_bayes.py`
  - **115.** Теорема Байеса
- `11_extras/09_bootstrap.py`
  - **116.** Bootstrap и доверительный интервал
- `11_extras/10_q_learning.py`
  - **117.** Q-learning - вкус Reinforcement Learning
- `11_extras/11_genetic_algorithm.py`
  - **118.** Генетический алгоритм
- `11_extras/12_simulated_annealing.py`
  - **119.** Simulated Annealing - метод отжига

## 12_probability_distributions — Распределения вероятностей

- `12_probability_distributions/01_discrete_basic.py`
  - **120.** Bernoulli(p)
  - **121.** Binomial(n, p)
  - **122.** Categorical / Multinoulli (k классов)
  - **123.** Multinomial(n, p1..pk)
  - **124.** Poisson(lambda)
  - **125.** Geometric(p)
  - **126.** Negative Binomial(r, p)
  - **127.** Hypergeometric(N, K, n)
  - **128.** Discrete Uniform
  - **129.** Zipf-распределение
- `12_probability_distributions/02_continuous_basic.py`
  - **130.** Uniform(a, b)
  - **131.** Normal/Gaussian(mu, sigma^2)
  - **132.** Exponential(lambda)
  - **133.** Laplace(mu, b)
  - **134.** Cauchy(x0, gamma)
  - **135.** LogNormal(mu, sigma)
  - **136.** Pareto(alpha, xm)
  - **137.** Weibull(k, lambda)
  - **138.** Gumbel
  - **139.** Rayleigh(sigma)
- `12_probability_distributions/03_continuous_advanced.py`
  - **140.** Beta(alpha, beta)
  - **141.** Gamma(k, theta)
  - **142.** Chi-Squared(k)
  - **143.** Student-t(nu)
  - **144.** F(d1, d2)
  - **145.** Dirichlet(alpha)
  - **146.** Wishart (концепт)
  - **147.** Inverse-Gamma (концепт)
- `12_probability_distributions/04_multivariate.py`
  - **148.** Многомерное нормальное MVN(mu, Sigma)
  - **149.** Форма ковариации
  - **150.** Расстояние Махаланобиса
  - **151.** Сэмплирование через Cholesky
  - **152.** Смесь гауссиан (GMM) и условное MVN

## 13_statistics_tests — Статистические тесты

- `13_statistics_tests/01_means_tests.py`
  - **153.** Одновыборочный t-тест
  - **154.** Двухвыборочный t-тест (равные дисперсии)
  - **155.** t-тест Уэлча (неравные дисперсии)
  - **156.** Парный t-тест
  - **157.** z-тест
- `13_statistics_tests/02_variance_tests.py`
  - **158.** F-тест отношения дисперсий
  - **159.** Тест Левена
  - **160.** Тест Бартлетта (концепт)
  - **161.** Однофакторная ANOVA
- `13_statistics_tests/03_nonparametric.py`
  - **162.** Mann-Whitney U
  - **163.** Wilcoxon signed-rank
  - **164.** Kruskal-Wallis (концепт)
  - **165.** Знаковый тест (sign test)
  - **166.** Тест Фридмана (концепт)
- `13_statistics_tests/04_distribution_tests.py`
  - **167.** Kolmogorov-Smirnov (одновыборочный)
  - **168.** Shapiro-Wilk (концепт)
  - **169.** Хи-квадрат критерий согласия
  - **170.** Хи-квадрат критерий независимости (таблицы сопряжённости)
- `13_statistics_tests/05_multiple_testing.py`
  - **171.** Поправка Бонферрони
  - **172.** Метод Холма
  - **173.** Benjamini-Hochberg (FDR)
  - **174.** Перестановочный тест
  - **175.** Бутстрап-тест
  - **176.** Тест МакНемара

## 14_distance_metrics — Метрики расстояний

- `14_distance_metrics/01_vector_distances.py`
  - **177.** Евклидово (L2)
  - **178.** Квадрат евклидова
  - **179.** Manhattan (L1)
  - **180.** Chebyshev (L_inf)
  - **181.** Minkowski-p
  - **182.** Canberra
  - **183.** Bray-Curtis
  - **184.** Cosine distance
  - **185.** Pearson distance
- `14_distance_metrics/02_set_string_distances.py`
  - **186.** Jaccard
  - **187.** Dice (Sorensen-Dice)
  - **188.** Hamming
  - **189.** Levenshtein (edit distance)
  - **190.** Dynamic Time Warping (DTW)
  - **191.** Haversine
- `14_distance_metrics/03_distribution_distances.py`
  - **192.** KL-дивергенция
  - **193.** Jensen-Shannon (JS)
  - **194.** Hellinger
  - **195.** Bhattacharyya
  - **196.** Wasserstein-1D (Earth Mover's)
  - **197.** Mahalanobis (как расстояние между точкой и распределением)

## 15_feature_engineering — Feature engineering

- `15_feature_engineering/01_numeric_transforms.py`
  - **198.** Log-преобразование
  - **199.** Sqrt
  - **200.** Обратное (reciprocal)
  - **201.** Box-Cox
  - **202.** Yeo-Johnson
  - **203.** Равномерное (equal-width) биннинг
  - **204.** Квантильный (equal-frequency) биннинг
  - **205.** Дискретизация в ординальное
- `15_feature_engineering/02_categorical_encodings.py`
  - **206.** One-Hot Encoding (recap)
  - **207.** Label Encoding
  - **208.** Ordinal Encoding
  - **209.** Target Encoding
  - **210.** Frequency Encoding
  - **211.** Leave-One-Out Encoding
  - **212.** Weight of Evidence (WoE)
  - **213.** Hashing trick
- `15_feature_engineering/03_text_features.py`
  - **214.** Tokenization
  - **215.** N-grams (word)
  - **216.** Char-grams
  - **217.** Bag-of-Words (BoW)
  - **218.** TF-IDF
  - **219.** HashingVectorizer
- `15_feature_engineering/04_datetime_features.py`
  - **220.** Календарные признаки
  - **221.** Циклическое sin/cos-кодирование
  - **222.** Флаг рабочего дня
  - **223.** Days-since-event
  - **224.** Простой holiday-флаг
  - **225.** Сезонность через циклы по дню года
- `15_feature_engineering/05_interactions.py`
  - **226.** Полиномиальные признаки
  - **227.** Попарные произведения
  - **228.** Отношения
  - **229.** Group-by агрегации

## 16_preprocessing — Препроцессинг

- `16_preprocessing/01_scalers.py`
  - **230.** StandardScaler
  - **231.** MinMaxScaler
  - **232.** MaxAbsScaler
  - **233.** RobustScaler
  - **234.** Normalizer L2
  - **235.** Normalizer L1
  - **236.** Binarizer
- `16_preprocessing/02_power_transforms.py`
  - **237.** QuantileTransformer
  - **238.** PowerTransformer (Yeo-Johnson)
  - **239.** KBinsDiscretizer
- `16_preprocessing/03_pipelines.py`
  - **240.** Pipeline
  - **241.** ColumnTransformer
  - **242.** FunctionTransformer
  - **243.** Data leakage из-за неправильного порядка
- `16_preprocessing/04_imputation.py`
  - **244.** SimpleImputer
  - **245.** KNNImputer
  - **246.** IterativeImputer (концепт)

## 17_linear_models_extended — Расширенные линейные модели

- `17_linear_models_extended/01_solvers.py`
  - **247.** Ridge — closed form vs SGD
  - **248.** Lasso через coordinate descent
  - **249.** ElasticNetCV
- `17_linear_models_extended/02_robust_regression.py`
  - **250.** Huber
  - **251.** RANSAC
  - **252.** Theil-Sen
  - **253.** Quantile regression
- `17_linear_models_extended/03_bayesian_linear.py`
  - **254.** BayesianRidge
  - **255.** ARD (Automatic Relevance Determination)
- `17_linear_models_extended/04_glm.py`
  - **256.** Poisson regression
  - **257.** Gamma regression
  - **258.** Tweedie regression
- `17_linear_models_extended/05_misc.py`
  - **259.** PassiveAggressive
  - **260.** SGDRegressor
  - **261.** OrthogonalMatchingPursuit (OMP)
  - **262.** LARS (Least Angle Regression)
  - **263.** Isotonic regression

## 18_svm_family — Семейство SVM

- `18_svm_family/01_hard_soft_margin.py`
  - **264.** Hard-margin SVM (концепт)
  - **265.** Soft-margin SVM с параметром C
  - **266.** Hinge vs squared hinge
- `18_svm_family/02_kernels.py`
  - **267.** Linear kernel
  - **268.** Polynomial kernel
  - **269.** RBF (Gaussian) kernel
  - **270.** Sigmoid kernel
- `18_svm_family/03_oneclass_nu.py`
  - **271.** NuSVC
  - **272.** NuSVR
  - **273.** OneClassSVM

## 19_trees_extended — Расширенные деревья

- `19_trees_extended/01_split_criteria.py`
  - **274.** Gini impurity
  - **275.** Entropy
  - **276.** MSE (variance) для регрессии
  - **277.** MAE для регрессии
  - **278.** Variance reduction
- `19_trees_extended/02_pruning.py`
  - **279.** Pre-pruning (предобрезка)
  - **280.** Cost-complexity pruning (post-pruning, alpha)
  - **281.** REP (Reduced Error Pruning, концепт)
- `19_trees_extended/03_variants.py`
  - **282.** ExtraTrees vs RandomForest
  - **283.** Регрессионное дерево
  - **284.** Oblique splits (косые расщепления, концепт)
  - **285.** Поверхностный взгляд на дерево

## 20_boosting_family — Бустинг

- `20_boosting_family/01_adaboost_variants.py`
  - **600.** AdaBoost — общая идея
  - **601.** AdaBoost SAMME (Stagewise Additive Modeling)
  - **602.** AdaBoost.M1
  - **603.** AdaBoost.R2 (идея регрессии)
  - **604.** Слабые ученики — pen' (decision stumps)
- `20_boosting_family/02_gradient_boosting.py`
  - **605.** Gradient Boosting — общая идея
  - **606.** GB для регрессии (MSE) — обучение на остатках
  - **607.** GB для классификации (log-loss) — концепт
  - **608.** Shrinkage (learning rate nu)
  - **609.** n_estimators и переобучение
- `20_boosting_family/03_xgboost_concepts.py`
  - **610.** Регуляризованная функция потерь XGBoost
  - **611.** Второй порядок (Newton step)
  - **612.** Выигрыш сплита (gain)
  - **613.** Псевдокод одного дерева (упрощённо)
  - **614.** Демонстрация — sklearn GradientBoostingClassifier как прокси
- `20_boosting_family/04_lightgbm_catboost.py`
  - **615.** Histogram-based splits
  - **616.** GOSS — Gradient-based One-Side Sampling
  - **617.** EFB — Exclusive Feature Bundling
  - **618.** Leaf-wise (best-first) рост дерева
  - **619.** CatBoost — Ordered Target Statistics
  - **620.** CatBoost — Ordered Boosting

## 21_clustering_extended — Расширенная кластеризация

- `21_clustering_extended/01_kmeans_variants.py`
  - **621.** Lloyd's K-Means
  - **622.** MiniBatch K-Means
  - **623.** K-Medoids (PAM — Partitioning Around Medoids)
  - **624.** Bisecting K-Means
- `21_clustering_extended/02_gmm_em.py`
  - **625.** Gaussian Mixture Model (GMM)
  - **626.** EM-алгоритм — E-шаг
  - **627.** EM-алгоритм — M-шаг
  - **628.** Лог-правдоподобие монотонно растёт
- `21_clustering_extended/03_density_clustering.py`
  - **629.** Mean-Shift
  - **630.** DBSCAN (recap)
  - **631.** OPTICS
  - **632.** HDBSCAN — концепт
- `21_clustering_extended/04_spectral_birch_affinity.py`
  - **633.** Spectral clustering
  - **634.** Affinity Propagation
  - **635.** BIRCH
- `21_clustering_extended/05_fuzzy_cmeans.py`
  - **636.** Fuzzy C-Means (FCM)
  - **637.** Обновление принадлежностей
  - **638.** Обновление центров
  - **639.** Целевая функция

## 22_dim_reduction_extended — Понижение размерности

- `22_dim_reduction_extended/01_lda_qda.py`
  - **640.** Linear Discriminant Analysis (LDA)
  - **641.** Within-class scatter S_W
  - **642.** Between-class scatter S_B
  - **643.** Размерность LDA-проекции
  - **644.** Quadratic Discriminant Analysis (QDA)
- `22_dim_reduction_extended/02_factor_models.py`
  - **645.** Factor Analysis
  - **646.** Independent Component Analysis (ICA)
  - **647.** Non-negative Matrix Factorization (NMF)
- `22_dim_reduction_extended/03_manifold.py`
  - **648.** MDS — Multidimensional Scaling
  - **649.** IsoMap
  - **650.** LLE — Locally Linear Embedding
- `22_dim_reduction_extended/04_random_projection.py`
  - **651.** Random Projection — идея
  - **652.** Gaussian Random Projection
  - **653.** Sparse Random Projection
  - **654.** Johnson-Lindenstrauss lemma

## 23_matrix_factorization — Матричная факторизация

- `23_matrix_factorization/01_svd_truncated.py`
  - **655.** SVD — Singular Value Decomposition
  - **656.** Truncated SVD (rank-k approximation)
  - **657.** Truncated SVD как LSA / LSI
  - **658.** Сжатие данных через rank-k
- `23_matrix_factorization/02_mf_for_cf.py`
  - **659.** Collaborative Filtering через матричное разложение
  - **660.** SGD-обновления Funk-SVD
  - **661.** ALS — Alternating Least Squares
  - **662.** Регуляризация L2 на P и Q
- `23_matrix_factorization/03_nmf_topics.py`
  - **663.** NMF на TF-IDF матрице
  - **664.** Топ-термины для темы
  - **665.** Документ как смесь тем
- `23_matrix_factorization/04_fm_concepts.py`
  - **666.** Factorization Machines (FM) — формула
  - **667.** Почему FM хороши для разреженных данных
  - **668.** Эффективное вычисление парного слагаемого

## 24_associations — Ассоциативные правила

- `24_associations/01_apriori.py`
  - **669.** Транзакции и itemset
  - **670.** Support (поддержка)
  - **671.** Apriori-принцип (anti-monotone)
  - **672.** Confidence и Lift для правила X => Y
- `24_associations/02_metrics.py`
  - **673.** Support
  - **674.** Confidence
  - **675.** Lift
  - **676.** Leverage
  - **677.** Conviction
  - **678.** All-confidence

## 25_anomaly_detection — Поиск аномалий

- `25_anomaly_detection/01_statistical.py`
  - **679.** Z-score
  - **680.** IQR-метод (boxplot rule)
  - **681.** MAD — Median Absolute Deviation
  - **682.** Modified Z-score
  - **683.** Mahalanobis distance
- `25_anomaly_detection/02_model_based.py`
  - **684.** Isolation Forest
  - **685.** LOF — Local Outlier Factor
  - **686.** Elliptic Envelope
  - **687.** One-Class SVM
- `25_anomaly_detection/03_reconstruction.py`
  - **688.** Autoencoder для аномалий — идея
  - **689.** Bottleneck
  - **690.** Скор аномальности

## 26_semi_supervised — Semi-supervised обучение

- `26_semi_supervised/01_self_training.py`
  - **691.** Semi-supervised learning — постановка
  - **692.** Self-training (pseudo-labeling)
  - **693.** Порог уверенности
  - **694.** Риск дрейфа ошибок
- `26_semi_supervised/02_label_propagation.py`
  - **695.** Граф сходства
  - **696.** Label Propagation
  - **697.** Label Spreading
- `26_semi_supervised/03_concepts.py`
  - **698.** Co-training
  - **699.** Tri-training
  - **700.** Mean Teacher
  - **701.** Π-model
  - **702.** FixMatch

## 27_self_supervised — Self-supervised обучение

- `27_self_supervised/01_contrastive_intro.py`
  - **703.** Self-supervised learning (SSL) — идея
  - **704.** Augmentation pairs (positive pair)
  - **705.** Contrastive learning
  - **706.** InfoNCE / NT-Xent loss
- `27_self_supervised/02_methods_concepts.py`
  - **707.** SimCLR
  - **708.** MoCo (Momentum Contrast)
  - **709.** BYOL (Bootstrap Your Own Latent)
  - **710.** SimSiam
  - **711.** Barlow Twins
- `27_self_supervised/03_masked_modeling.py`
  - **712.** Masked Language Modeling (MLM)
  - **713.** Masked Autoencoder (MAE)
  - **714.** Цель: восстановить скрытое по видимому

## 28_neural_layers — Слои нейросетей

- `28_neural_layers/01_dense_embedding.py`
  - **280.** Linear / Dense layer
  - **281.** Bias
  - **282.** Embedding layer
  - **283.** Flatten / Reshape
- `28_neural_layers/02_conv_variants.py`
  - **284.** Conv1D вручную
  - **285.** Conv2D вручную
  - **286.** Padding "valid" vs "same"
  - **287.** Strided conv
  - **288.** Dilated / atrous conv
- `28_neural_layers/03_special_convs.py`
  - **289.** ConvTranspose / Upsampling
  - **290.** Depthwise convolution
  - **291.** Pointwise (1x1) conv
  - **292.** Separable convolution
- `28_neural_layers/04_pooling_layers.py`
  - **293.** MaxPool 2x2
  - **294.** AvgPool 2x2
  - **295.** Global Average Pooling (GAP)
  - **296.** Adaptive pooling
- `28_neural_layers/05_dropout_variants.py`
  - **297.** Vanilla Dropout
  - **298.** SpatialDropout (Drop channels)
  - **299.** AlphaDropout
  - **300.** Scheduled dropout

## 29_activations_extended — Функции активации

- `29_activations_extended/01_relu_family.py`
  - **301.** ReLU(x) = max(0, x)
  - **302.** LeakyReLU(x) = x if x>0 else alpha*x
  - **303.** PReLU
  - **304.** ELU(x) = x if x>0 else alpha*(exp(x)-1)
  - **305.** CELU
  - **306.** SELU = lambda * ELU(alpha)
  - **307.** GELU(x) ≈ x * Phi(x)
  - **308.** Swish / SiLU(x) = x * sigmoid(x)
  - **309.** Mish(x) = x * tanh(softplus(x))
  - **310.** Softplus(x) = log(1+exp(x))
  - **311.** Softsign(x) = x / (1+|x|)
  - **312.** HardSigmoid / HardTanh / HardSwish
- `29_activations_extended/02_glu_family.py`
  - **313.** Maxout
  - **314.** GLU (Gated Linear Unit)
  - **315.** ReGLU = a * ReLU(b)
  - **316.** SwiGLU = a * SiLU(b) = a * (b * sigmoid(b))
  - **317.** GeGLU = a * GELU(b)
- `29_activations_extended/03_output_activations.py`
  - **318.** Sigmoid
  - **319.** Softmax с трюком стабильности
  - **320.** Log-Softmax
  - **321.** Sparsemax (концепт)
  - **322.** Hierarchical softmax (концепт)

## 30_optimizers_extended — Оптимизаторы

- `30_optimizers_extended/01_sgd_momentum.py`
  - **323.** Vanilla SGD
  - **324.** Momentum / Heavy ball
  - **325.** Nesterov Accelerated Gradient
- `30_optimizers_extended/02_adaptive.py`
  - **326.** Adagrad
  - **327.** RMSProp
  - **328.** AdaDelta
  - **329.** Adam = Momentum + RMSProp + bias correction
  - **330.** AMSGrad
  - **331.** AdamW
- `30_optimizers_extended/03_modern.py`
  - **332.** AdaMax
  - **333.** Nadam
  - **334.** RAdam
  - **335.** Lion (EvoLved Sign Momentum)
  - **336.** LAMB
  - **337.** LARS
- `30_optimizers_extended/04_meta_optimizers.py`
  - **338.** Lookahead
  - **339.** SWA (Stochastic Weight Averaging)
  - **340.** Polyak averaging (EMA of weights)
  - **341.** Gradient accumulation

## 31_losses_extended — Функции потерь

- `31_losses_extended/01_regression_losses.py`
  - **342.** MSE = mean((y-yhat)^2)
  - **343.** MAE = mean(|y-yhat|)
  - **344.** Huber loss
  - **345.** Quantile / Pinball loss
  - **346.** Log-cosh
- `31_losses_extended/02_classification_losses.py`
  - **347.** BCE (Binary Cross-Entropy)
  - **348.** CE (Categorical Cross-Entropy)
  - **349.** Focal loss
  - **350.** Label-smoothing CE
- `31_losses_extended/03_segmentation_losses.py`
  - **351.** Dice loss = 1 - 2|A ∩ B| / (|A|+|B|)
  - **352.** Tversky loss
  - **353.** IoU (Jaccard) loss
  - **354.** BCE+Dice combo
- `31_losses_extended/04_metric_losses.py`
  - **355.** Triplet loss
  - **356.** Contrastive loss (Hadsell et al.)
  - **357.** InfoNCE / NT-Xent
  - **358.** Cosine embedding loss

## 32_regularization_extended — Расширенная регуляризация

- `32_regularization_extended/01_weight_regs.py`
  - **359.** L1
  - **360.** L2 (Ridge / weight decay)
  - **361.** ElasticNet
  - **362.** MaxNorm
  - **363.** Orthogonal regularization
  - **364.** Spectral norm (power iteration)
- `32_regularization_extended/02_input_regs.py`
  - **365.** Label smoothing
  - **366.** MixUp
  - **367.** CutMix
  - **368.** CutOut
  - **369.** RandAugment (концепт)
- `32_regularization_extended/03_arch_regs.py`
  - **370.** Stochastic depth
  - **371.** DropConnect
  - **372.** DropBlock
  - **373.** Zoneout (RNN dropout)
  - **374.** R-Drop (концепт)
  - **375.** SAM — Sharpness-Aware Minimization (концепт)

## 33_init_extended — Инициализация весов

- `33_init_extended/01_basic_inits.py`
  - **376.** Zeros init
  - **377.** Ones init
  - **378.** Constant init
  - **379.** Uniform(a, b)
  - **380.** Normal(mu, sigma)
  - **381.** TruncatedNormal (rejection)
- `33_init_extended/02_smart_inits.py`
  - **382.** Xavier (Glorot) Uniform
  - **383.** Xavier Normal
  - **384.** He Uniform
  - **385.** He Normal
  - **386.** LeCun
  - **387.** Orthogonal init
  - **388.** Identity init
  - **389.** Variance scaling

## 34_normalization — Нормализация

- `34_normalization/01_norm_layers.py`
  - **390.** BatchNorm
  - **391.** LayerNorm
  - **392.** InstanceNorm
  - **393.** GroupNorm
- `34_normalization/02_modern_norms.py`
  - **394.** RMSNorm
  - **395.** WeightNorm
  - **396.** Spectral Normalization (power iteration)
  - **397.** ActNorm
  - **398.** Batch Renorm (концепт)
  - **399.** Filter Response Norm (FRN, концепт)

## 35_cnn_archs — Архитектуры CNN

- `35_cnn_archs/01_lenet_alexnet.py`
  - — LeNet-5 (LeCun, 1998)
  - — AlexNet (Krizhevsky, 2012)
  - — расчёт размеров после Conv
- `35_cnn_archs/02_vgg_inception.py`
  - — VGG (Simonyan, 2014)
  - — NiN (Network in Network)
  - — Inception module (GoogLeNet, 2014)
  - — вспомогательные классификаторы (auxiliary heads)
- `35_cnn_archs/03_resnet_family.py`
  - — ResNet (He, 2015) и skip connection
  - — ResNeXt (cardinality)
  - — DenseNet
  - — Wide ResNet
  - — gradient flow через skip
- `35_cnn_archs/04_efficient_archs.py`
  - — MobileNet — depthwise separable conv
  - — MobileNetV2 — inverted residual + linear bottleneck
  - — ShuffleNet — group conv + channel shuffle
  - — SqueezeNet — fire module
  - — EfficientNet — compound scaling
- `35_cnn_archs/05_attention_blocks.py`
  - — Squeeze-and-Excitation (SE) block
  - — CBAM (Convolutional Block Attention Module)
  - — ConvNeXt
  - — Vision Transformer (ViT)

## 36_rnn_extended — Расширенные RNN

- `36_rnn_extended/01_rnn_variants.py`
  - — Vanilla RNN
  - — LSTM (стандартный)
  - — Peephole LSTM
  - — GRU
  - — MGU (Minimal Gated Unit)
- `36_rnn_extended/02_bidir_stacked.py`
  - — Bidirectional RNN
  - — Stacked / multi-layer RNN
  - — residual connection между слоями RNN
- `36_rnn_extended/03_seq2seq_decoding.py`
  - — Encoder-Decoder (Seq2Seq)
  - — Greedy decoding
  - — Beam search
  - — Length normalization
  - — Top-k sampling
  - — Top-p (nucleus) sampling

## 37_attention — Механизмы attention

- `37_attention/01_attention_variants.py`
  - — Additive (Bahdanau) attention
  - — Multiplicative (Luong) attention
  - — Scaled dot-product attention
  - — матрица внимания
- `37_attention/02_multihead.py`
  - — Multi-head attention
  - — split/concat по головам
  - — параметры MHA
- `37_attention/03_self_cross.py`
  - — Self-attention
  - — Cross-attention
  - — Causal (look-ahead) mask
  - — Padding mask
- `37_attention/04_efficient_attention.py`
  - — Sliding-window attention (Longformer)
  - — Sparse attention (Sparse Transformer, BigBird)
  - — Linformer / Performer — линейное внимание
  - — FlashAttention
- `37_attention/05_positional_encodings.py`
  - — Sinusoidal positional encoding (Vaswani, 2017)
  - — Learned positional embedding
  - — Relative positional encoding (Shaw, T5)
  - — RoPE (Rotary Position Embedding)
  - — ALiBi (Attention with Linear Biases)

## 38_transformers — Архитектура Transformer

- `38_transformers/01_blocks.py`
  - — Transformer encoder block (Vaswani, 2017)
  - — LayerNorm
  - — FFN (Feed-Forward Network)
  - — Add (residual)
- `38_transformers/02_decoder_block.py`
  - — Decoder block
  - — Masked self-attention
  - — Cross-attention (encoder-decoder)
- `38_transformers/03_prenorm_postnorm.py`
  - — Post-LN (оригинальный, Vaswani 2017)
  - — Pre-LN (современный, GPT-2 и далее)
  - — эффект на масштаб активаций
- `38_transformers/04_variants.py`
  - — Weight tying
  - — KV cache (декодирование)
  - — Parallel residual (GPT-J)
  - — Mixture of Experts (MoE)
- `38_transformers/05_objectives.py`
  - — Causal LM (GPT)
  - — Masked LM (BERT)
  - — Span corruption (T5)
  - — Replaced token detection (ELECTRA)
  - — Denoising (BART)

## 39_nlp_classic — Классический NLP

- `39_nlp_classic/01_text_basics.py`
  - — токенизация по пробелам и regex
  - — нормализация (lowercasing)
  - — stopwords
  - — stemming (Porter)
  - — лемматизация
- `39_nlp_classic/02_ngrams_bow.py`
  - — word n-grams
  - — character n-grams
  - — Bag-of-Words
  - — нормализация BoW
- `39_nlp_classic/03_tfidf_lsa.py`
  - — TF (term frequency)
  - — IDF (inverse document frequency)
  - — TF-IDF
  - — LSA (Latent Semantic Analysis)
- `39_nlp_classic/04_topic_models.py`
  - — LDA (Latent Dirichlet Allocation)
  - — Dirichlet prior
  - — интерпретация
- `39_nlp_classic/05_seq_labeling.py`
  - — POS-tagging
  - — HMM (Hidden Markov Model)
  - — алгоритм Витерби
  - — CRF (Conditional Random Field)

## 40_nlp_embeddings — Word embeddings

- `40_nlp_embeddings/01_word2vec.py`
  - — Skip-gram
  - — Negative sampling
  - — косинусное сходство
- `40_nlp_embeddings/02_glove_fasttext.py`
  - — GloVe
  - — FastText (subword)
- `40_nlp_embeddings/03_sentence_embeddings.py`
  - — mean/max pooling
  - — SIF (Smooth Inverse Frequency)
  - — Doc2Vec (PV-DM, PV-DBOW)

## 41_lm_pretrained — Предобученные языковые модели

- `41_lm_pretrained/01_objectives.py`
  - — Autoregressive LM (GPT)
  - — Perplexity
  - — MLM (BERT)
  - — NSP (Next Sentence Prediction)
  - — SOP (Sentence Order Prediction, ALBERT)
- `41_lm_pretrained/02_finetuning.py`
  - — feature extraction
  - — full fine-tuning
  - — discriminative LR / layer-wise LR decay
  - — classifier head
  - — ELECTRA replaced-token detection
- `41_lm_pretrained/03_alignment.py`
  - — Instruction tuning (SFT)
  - — Reward model
  - — RLHF (PPO)
  - — DPO (Direct Preference Optimization)
- `41_lm_pretrained/04_inference.py`
  - — Greedy decoding
  - — Sampling с температурой T
  - — Top-k sampling
  - — Top-p (nucleus) sampling
  - — repetition penalty

## 42_cv_basics — Основы computer vision

- `42_cv_basics/01_image_repr.py`
  - — Изображение как ndarray
  - — RGB <-> grayscale
  - — HSV
  - — Гамма-коррекция
  - — Гистограмма
  - — Эквализация гистограммы
- `42_cv_basics/02_conv_props.py`
  - — Линейность свёртки
  - — Shift-equivariance
  - — Режимы padding
  - — Receptive field (рецептивное поле)
  - — Stride и dilation
- `42_cv_basics/03_speedups.py`
  - — im2col
  - — Свёртка через matmul
  - — Свёртка через FFT
  - — Сепарабельные фильтры

## 43_cv_classic — Классический computer vision

- `43_cv_classic/01_filters.py`
  - — Box blur
  - — Gaussian blur
  - — Median filter
  - — Bilateral filter
- `43_cv_classic/02_edges.py`
  - — Sobel X/Y
  - — Prewitt
  - — Scharr
  - — Laplacian
  - — Canny pipeline
- `43_cv_classic/03_corners_features.py`
  - — Harris corner detector
  - — FAST corner
  - — ORB (Oriented FAST + Rotated BRIEF)
  - — SIFT
  - — SURF
- `43_cv_classic/04_classical_descriptors.py`
  - — HOG (Histogram of Oriented Gradients)
  - — LBP (Local Binary Pattern)
  - — Haar-like features
  - — Integral image
  - — Viola-Jones cascade

## 44_cv_detection — Детекция объектов

- `44_cv_detection/01_sliding_window.py`
  - — Sliding window
  - — Template matching
  - — Image pyramid
- `44_cv_detection/02_iou_nms.py`
  - — IoU (Intersection over Union)
  - — NMS (Non-Maximum Suppression)
  - — Soft-NMS
  - — GIoU / DIoU / CIoU
- `44_cv_detection/03_rcnn_family.py`
  - — R-CNN
  - — Fast R-CNN
  - — Faster R-CNN
  - — Mask R-CNN
  - — RoI Pooling vs RoI Align
- `44_cv_detection/04_yolo_ssd.py`
  - — YOLO (You Only Look Once)
  - — Anchor boxes
  - — SSD (Single Shot Detector)
  - — FPN (Feature Pyramid Network)
  - — Focal Loss

## 45_cv_segmentation — Сегментация изображений

- `45_cv_segmentation/01_semantic_seg.py`
  - — Semantic segmentation
  - — FCN (Fully Convolutional Network)
  - — U-Net
  - — SegNet
  - — DeepLab + ASPP
- `45_cv_segmentation/02_instance_panoptic.py`
  - — Semantic vs Instance vs Panoptic
  - — Mask R-CNN для instance
  - — Panoptic Segmentation
  - — PQ метрика (Panoptic Quality)
- `45_cv_segmentation/03_seg_losses.py`
  - — Dice loss
  - — BCE + Dice
  - — Tversky loss
  - — Lovasz loss
  - — CRF post-processing

## 46_generative_classic — Классические генеративные модели

- `46_generative_classic/01_simple_generation.py`
  - — Sampling из эмпирической гистограммы
  - — GMM sampling
  - — Autoregressive Bernoulli
  - — MLE-based generation
- `46_generative_classic/02_vae.py`
  - — VAE
  - — ELBO (Evidence Lower BOund)
  - — Reparameterization trick
  - — KL для N(mu, sigma) || N(0, I)
  - — Decoder
- `46_generative_classic/03_gan_basics.py`
  - — GAN
  - — Non-saturating loss for G
  - — Mode collapse
  - — Training dynamics
- `46_generative_classic/04_gan_variants.py`
  - — WGAN (Wasserstein GAN)
  - — WGAN-GP (Gradient Penalty)
  - — LSGAN
  - — Conditional GAN (cGAN)
  - — Pix2Pix
  - — CycleGAN

## 47_generative_modern — Современные генеративные модели

- `47_generative_modern/01_autoregressive.py`
  - — AR-модели
  - — PixelRNN
  - — PixelCNN
  - — WaveNet
  - — Causal masking
- `47_generative_modern/02_flows.py`
  - — Change of variables
  - — RealNVP coupling layer
  - — Glow 1x1 invertible conv
  - — Log-likelihood обучения
  - — Sampling
- `47_generative_modern/03_diffusion.py`
  - — Forward diffusion
  - — Closed-form для x_t из x_0
  - — Reverse process
  - — Denoising step
  - — Noise schedule
- `47_generative_modern/04_diffusion_extras.py`
  - — DDIM (Denoising Diffusion Implicit Models)
  - — Classifier-free guidance
  - — Latent Diffusion / Stable Diffusion
  - — Score-based models
  - — Consistency models

## 48_rl_extended — Reinforcement learning

- `48_rl_extended/01_mdp_dp.py`
  - — MDP (Markov Decision Process)
  - — Bellman expectation
  - — Bellman optimality
  - — Value iteration
  - — Policy iteration
- `48_rl_extended/02_mc_td.py`
  - — Monte Carlo prediction
  - — TD(0)
  - — n-step TD
  - — TD(lambda) и eligibility traces
  - — Bias-variance
- `48_rl_extended/03_q_learning_family.py`
  - — SARSA
  - — Expected SARSA
  - — Q-learning
  - — Double Q-learning
- `48_rl_extended/04_policy_gradient.py`
  - — REINFORCE (Monte-Carlo PG)
  - — REINFORCE with baseline
  - — Actor-Critic
  - — A2C / A3C
  - — GAE (Generalized Advantage Estimation)
- `48_rl_extended/05_deep_rl_concepts.py`
  - — DQN (Deep Q-Network)
  - — Double DQN
  - — Dueling DQN
  - — Rainbow
  - — PPO (Proximal Policy Optimization)
  - — DDPG, TD3, SAC
  - — Exploration

## 49_mlops_basics — Основы MLOps

- `49_mlops_basics/01_reproducibility.py`
  - — Случайность в ML
  - — Seed для numpy
  - — Seed для python random
  - — random_state в sklearn
  - — Версионирование окружения
  - — Источники невоспроизводимости
- `49_mlops_basics/02_data_splits.py`
  - — Train / Validation / Test
  - — Случайное разбиение
  - — Стратифицированное разбиение
  - — Временное разбиение (time-based)
  - — Group-aware split
  - — Holdout vs Cross-validation
- `49_mlops_basics/03_leakage.py`
  - — Что такое leakage
  - — Target leakage
  - — Train-test contamination
  - — Future leakage
  - — Как защищаться
- `49_mlops_basics/04_drift.py`
  - — Concept drift vs Data drift
  - — Label drift (prior shift)
  - — Population Stability Index (PSI)
  - — Kolmogorov-Smirnov для дрейфа
  - — Detect & React

## 50_interpretability — Интерпретируемость моделей

- `50_interpretability/01_global.py`
  - — Глобальная vs локальная интерпретация
  - — Коэффициенты линейной модели
  - — feature_importances_ в Random Forest
  - — Permutation importance
  - — Когда они расходятся
- `50_interpretability/02_partial_dependence.py`
  - — Partial Dependence Plot (PDP)
  - — Достоинства и недостатки PDP
  - — ICE (Individual Conditional Expectation)
  - — ALE (Accumulated Local Effects)
  - — Применение
- `50_interpretability/03_local_explanations.py`
  - — Локальная интерпретация
  - — LIME (Local Interpretable Model-agnostic Explanations)
  - — Веса близости
  - — SHAP (SHapley Additive exPlanations)
  - — KernelSHAP
- `50_interpretability/04_attribution.py`
  - — Saliency map
  - — Integrated Gradients (IG)
  - — Baseline
  - — Grad-CAM (для CNN)
  - — Аксиомы attribution

## 51_graph_ml — Graph ML

- `51_graph_ml/01_graphs_basics.py`
  - — Матрица смежности A
  - — Степень вершины и матрица степеней D
  - — Лапласиан L = D - A
  - — Нормализованный Лапласиан
  - — BFS (поиск в ширину)
  - — DFS (поиск в глубину)
- `51_graph_ml/02_pagerank_propagation.py`
  - — PageRank
  - — Power iteration для PageRank
  - — Personalized PageRank (PPR)
  - — Label Propagation на графе
  - — Сходимость
- `51_graph_ml/03_embeddings.py`
  - — Зачем эмбеддинги вершин
  - — DeepWalk
  - — node2vec — смещённые блуждания
  - — LINE / факторизация смежности
  - — Использование
- `51_graph_ml/04_gnn_concepts.py`
  - — Message Passing Framework
  - — GCN (Graph Convolutional Network)
  - — GraphSAGE
  - — GAT (Graph Attention Network)
  - — Многослойные GNN

## 52_time_series — Временные ряды

- `52_time_series/01_decomposition.py`
  - — Компоненты ряда
  - — Тренд через moving average
  - — Сезонность
  - — Остаток
  - — STL
- `52_time_series/02_stationarity.py`
  - — Стационарность (строгая и слабая)
  - — Дифференцирование
  - — Лог-преобразование
  - — ADF (Augmented Dickey-Fuller)
  - — KPSS
- `52_time_series/03_acf_pacf.py`
  - — Автокорреляция (ACF)
  - — Частичная автокорреляция (PACF)
  - — Использование ACF/PACF
  - — Доверительный интервал
  - — AR(1) ряд для демо
- `52_time_series/04_arima.py`
  - — AR(p) — авторегрессия
  - — MA(q) — скользящее среднее
  - — ARMA(p, q)
  - — ARIMA(p, d, q)
  - — SARIMA(p,d,q)(P,D,Q,s)
- `52_time_series/05_smoothing.py`
  - — Simple Exponential Smoothing (SES)
  - — Holt's linear trend
  - — Holt-Winters (аддитивная сезонность)
  - — Мультипликативная версия
  - — ETS notation

## 53_recommender — Рекомендательные системы

- `53_recommender/01_baselines.py`
  - — Матрица user-item
  - — Popularity baseline
  - — Most-similar-user
  - — Most-similar-item
  - — Cold start
- `53_recommender/02_cf_methods.py`
  - — User-user CF
  - — Item-item CF
  - — Меры сходства
  - — Уход в среднее
  - — Sparse-tricks
- `53_recommender/03_mf.py`
  - — Идея MF
  - — Целевая функция
  - — SGD-обучение
  - — Bias-члены
  - — Применение и расширения
- `53_recommender/04_evaluation.py`
  - — Precision@K
  - — Recall@K
  - — MAP@K (Mean Average Precision)
  - — NDCG@K
  - — MRR (Mean Reciprocal Rank)
  - — Coverage и Diversity

## 54_bayesian_ml — Байесовский ML

- `54_bayesian_ml/01_priors.py`
  - — Теорема Байеса
  - — Bayes для монеты с Beta-prior
  - — Сопряжённое (conjugate) prior
  - — Posterior mean vs MAP
  - — Credible interval
- `54_bayesian_ml/02_conjugate.py`
  - — Beta-Binomial
  - — Gamma-Poisson
  - — Normal-Normal (известная sigma^2)
  - — Normal-Inverse-Gamma (неизвестные mu и sigma^2)
  - — Зачем conjugate
- `54_bayesian_ml/03_bayesian_linear.py`
  - — Модель
  - — Закрытая форма posterior (Gaussian prior + Gaussian likelihood)
  - — Связь с Ridge
  - — Posterior predictive
  - — Зачем это нужно
- `54_bayesian_ml/04_gp.py`
  - — Что такое GP
  - — RBF (Gaussian) kernel
  - — GP regression posterior
  - — Сложность
  - — Применение

## 55_mcmc — MCMC и вариационный вывод

- `55_mcmc/01_simple_samplers.py`
  - — Inverse CDF sampling
  - — Rejection sampling
  - — Importance sampling
  - — Effective sample size (ESS)
  - — Когда что использовать
- `55_mcmc/02_mh_gibbs.py`
  - — Markov Chain Monte Carlo
  - — Random-walk Metropolis-Hastings
  - — Выбор шага
  - — Gibbs sampling
  - — Диагностика сходимости
- `55_mcmc/03_advanced.py`
  - — Slice sampling
  - — Hamiltonian Monte Carlo (HMC)
  - — NUTS (No U-Turn Sampler)
  - — Parallel tempering
  - — Variational Inference (VI)

## 56_optimization_extended — Расширенная оптимизация

- `56_optimization_extended/01_convex.py`
  - — Выпуклое множество
  - — Выпуклая функция
  - — Проверка выпуклости вдоль прямой (1D-сужение)
  - — Сильная выпуклость
  - — Неравенство Йенсена
- `56_optimization_extended/02_first_order.py`
  - — Градиентный спуск (GD)
  - — Метод Нестерова (NAG)
  - — Heavy ball (Polyak momentum)
  - — Субградиентный метод
  - — Proximal оператор + soft-thresholding для L1
- `56_optimization_extended/03_coordinate.py`
  - — Координатный спуск (CD)
  - — Циклический CD
  - — Случайный CD
  - — Блочный CD (BCD)
- `56_optimization_extended/04_second_order.py`
  - — Метод Ньютона
  - — Квази-Ньютон (BFGS)
  - — L-BFGS
  - — Метод сопряженных градиентов (CG)
- `56_optimization_extended/05_constrained.py`
  - — Лагранжиан
  - — Условия KKT
  - — Проектированный градиент (projected GD)
  - — ADMM
  - — FISTA

## 57_hpo_extended — Подбор гиперпараметров

- `57_hpo_extended/01_search_methods.py`
  - — Grid Search
  - — Random Search
  - — Почему random выигрывает при низкой effective dimensionality
- `57_hpo_extended/02_bayesopt.py`
  - — GP-суррогат
  - — Expected Improvement (EI)
  - — Upper Confidence Bound (UCB / LCB)
  - — Probability of Improvement (PI)
  - — Цикл BO
- `57_hpo_extended/03_tpe_others.py`
  - — TPE (Tree-structured Parzen Estimator)
  - — CMA-ES (Covariance Matrix Adaptation)
  - — PBT (Population-Based Training)
- `57_hpo_extended/04_multifidelity.py`
  - — Successive Halving (SHA)
  - — Hyperband
  - — ASHA (Asynchronous Successive Halving)
  - — BOHB

## 58_info_theory — Теория информации

- `58_info_theory/01_entropy.py`
  - — Энтропия Шеннона
  - — Совместная энтропия
  - — Условная энтропия
  - — Взаимная информация (MI)
- `58_info_theory/02_divergences.py`
  - — KL-дивергенция
  - — Дивергенция Йенсена-Шеннона (JS)
  - — Хеллингер
  - — Total Variation (TV)
  - — Бхаттачарья
- `58_info_theory/03_advanced.py`
  - — Информация Фишера
  - — Связь Cross-Entropy <-> KL
  - — Information Bottleneck (IB)
  - — MIC (Maximal Information Coefficient)
  - — MINE (Mutual Information Neural Estimation)

## 59_linalg_extended — Расширенная линейная алгебра

- `59_linalg_extended/01_decompositions.py`
  - — LU-разложение
  - — QR-разложение
  - — Cholesky
  - — Eigendecomposition
  - — Schur-разложение
- `59_linalg_extended/02_pseudoinverse.py`
  - — Moore-Penrose pseudoinverse
  - — МНК через псевдообратную
  - — Число обусловленности
  - — Rank-revealing разложения
- `59_linalg_extended/03_iterative_methods.py`
  - — Power iteration
  - — Krylov subspace
  - — Lanczos
  - — Arnoldi
  - — CG как Krylov-метод

## 60_calculus_extended — Расширенный матанализ

- `60_calculus_extended/01_multivar.py`
  - — Частные производные
  - — Градиент
  - — Якобиан
  - — Гессиан
  - — Дивергенция и ротор (концепт)
- `60_calculus_extended/02_advanced.py`
  - — Многомерное правило цепочки
  - — Теорема о неявной функции (концепт)
  - — Ряд Тейлора (1D и 2D)
  - — Множители Лагранжа
- `60_calculus_extended/03_autodiff.py`
  - — Forward-mode AD через dual numbers
  - — Reverse-mode AD
  - — VJP (Vector-Jacobian product)

## 61_probability_extended — Расширенная теория вероятностей

- `61_probability_extended/01_concepts.py`
  - — Пространство элементарных исходов
  - — Сигма-алгебра (интуиция)
  - — Независимость
  - — Условная вероятность
  - — Формула полной вероятности
  - — Теорема Байеса
- `61_probability_extended/02_moments.py`
  - — Линейность матожидания
  - — Дисперсия
  - — Ковариация и корреляция
  - — Высшие моменты
  - — Производящая функция моментов (MGF)
  - — Характеристическая функция
- `61_probability_extended/03_inequalities.py`
  - — Неравенство Маркова
  - — Неравенство Чебышева
  - — Неравенство Йенсена
  - — Неравенство Хоффдинга
- `61_probability_extended/04_markov_chains.py`
  - — Матрица переходов
  - — n-шаговая переходная матрица
  - — Стационарное распределение
  - — Эргодичность

## 62_tricks_training — Трюки обучения

- `62_tricks_training/01_precision_memory.py`
  - — Mixed precision (FP16/BF16)
  - — Gradient checkpointing
  - — Gradient accumulation
- `62_tricks_training/02_grad_tricks.py`
  - — Gradient clipping by norm
  - — Gradient clipping by value
  - — Gradient noise
  - — Lookahead optimizer
- `62_tricks_training/03_lr_schedules.py`
  - — Warmup
  - — Constant
  - — Step decay
  - — Exponential decay
  - — Cosine annealing
  - — Cosine with restarts (SGDR)
  - — ReduceLROnPlateau
- `62_tricks_training/04_weight_averaging.py`
  - — EMA (Exponential Moving Average)
  - — SWA (Stochastic Weight Averaging)
  - — Polyak averaging
- `62_tricks_training/05_compression.py`
  - — Knowledge distillation
  - — Magnitude pruning
  - — Structured pruning
  - — PTQ (Post-Training Quantization)
  - — QAT (Quantization-Aware Training)

## 63_data_augmentation — Аугментация данных

- `63_data_augmentation/01_image_augs.py`
  - — Flip H/V
  - — Rotate 90/180/270
  - — Random crop
  - — Color jitter
  - — Gaussian blur
  - — Cutout
- `63_data_augmentation/02_image_modern.py`
  - — MixUp
  - — CutMix
  - — RandAugment
  - — AugMix
- `63_data_augmentation/03_text_audio_tabular.py`
  - — Text — synonym replacement
  - — Text — random deletion / swap
  - — Text — back-translation
  - — Audio — time stretch
  - — Audio — pitch shift
  - — Audio — SpecAugment
  - — Tabular — SMOTE
  - — Tabular — ADASYN

## 64_meta_learning — Meta-learning

- `64_meta_learning/01_transfer.py`
  - — Transfer learning
  - — Feature extraction (заморозка энкодера)
  - — Fine-tuning
  - — Domain adaptation
- `64_meta_learning/02_few_shot.py`
  - — N-way K-shot setup
  - — Prototypical Networks (Snell et al., 2017)
  - — Matching Networks (Vinyals et al., 2016)
  - — MAML (Model-Agnostic Meta-Learning, Finn et al., 2017)
  - — Reptile (Nichol et al., 2018)

## 65_metric_learning — Metric learning

- `65_metric_learning/01_siamese_triplet.py`
  - — Siamese networks
  - — Contrastive loss (Hadsell, 2006)
  - — Triplet loss (FaceNet)
  - — Easy / Hard / Semi-hard negatives
  - — Semi-hard mining (стратегия)
- `65_metric_learning/02_face_recognition.py`
  - — Center loss (Wen et al., 2016)
  - — Large-Margin Softmax (L-Softmax)
  - — SphereFace (A-Softmax)
  - — CosFace (Large Margin Cosine Loss)
  - — ArcFace (Additive Angular Margin Loss)

## 66_active_learning — Active learning

- `66_active_learning/01_query_strategies.py`
  - — Active Learning
  - — Uncertainty sampling (least confidence)
  - — Margin sampling
  - — Entropy sampling
- `66_active_learning/02_advanced.py`
  - — Query-by-committee (QBC)
  - — Expected Model Change
  - — Expected Error Reduction (EER)
  - — Density-Weighted methods
  - — BatchBALD (Bayesian Active Learning by Disagreement, batch версия)

## 67_online_learning — Online learning

- `67_online_learning/01_online_sgd.py`
  - — Online learning
  - — Online Perceptron (Rosenblatt, 1958)
  - — Online SGD
  - — Passive-Aggressive (PA, Crammer 2006)
  - — FTRL (Follow-The-Regularized-Leader, McMahan 2013)
  - — Hoeffding Trees (VFDT, концепт)
- `67_online_learning/02_bandits.py`
  - — Multi-armed bandit (MAB)
  - — Epsilon-greedy
  - — UCB1 (Auer 2002)
  - — Thompson sampling (Bernoulli arms)
  - — Contextual bandits (LinUCB)
- `67_online_learning/03_regret.py`
  - — Regret
  - — Cumulative regret plot
  - — Exploration vs Exploitation
  - — Theoretical bounds

## 68_kernel_methods — Kernel methods

- `68_kernel_methods/01_kernel_recap.py`
  - — Kernel trick
  - — Linear kernel
  - — Polynomial kernel
  - — RBF (Gaussian) kernel
  - — Laplace kernel
  - — Sigmoid kernel
  - — String kernel (концепт)
  - — Gram matrix
- `68_kernel_methods/02_kernel_models.py`
  - — Kernel Ridge Regression (KRR)
  - — Kernel PCA
  - — Kernel K-means
- `68_kernel_methods/03_approximations.py`
  - — Зачем приближать?
  - — Nyström approximation
  - — Random Fourier Features (RFF) для RBF
  - — Compare approximation vs true

## 69_density_estimation — Density estimation

- `69_density_estimation/01_histogram_kde.py`
  - — Histogram density
  - — Kernel Density Estimation (KDE)
  - — Выбор bandwidth
  - — Silverman's rule of thumb
  - — Scott's rule
  - — Multivariate KDE
- `69_density_estimation/02_gmm_density.py`
  - — GMM как density model
  - — Score samples
  - — Order selection (сколько компонент K выбрать?)
  - — BIC и AIC
- `69_density_estimation/03_density_ratio.py`
  - — Density ratio estimation (DRE)
  - — DRE через классификацию
  - — KL и JS через DRE
  - — Copula (концепт)

## 70_causal_inference — Causal inference

- `70_causal_inference/01_correlation_causation.py`
  - — Correlation != causation
  - — Confounder
  - — Partial correlation
  - — Backdoor adjustment
- `70_causal_inference/02_rcts_obs.py`
  - — RCT (Randomized Controlled Trial)
  - — ATE (Average Treatment Effect)
  - — Observational studies
  - — Simpson's paradox
- `70_causal_inference/03_methods.py`
  - — Propensity score
  - — IPW (Inverse Propensity Weighting)
  - — Matching (1-NN на propensity score)
  - — Difference-in-Differences (DiD)
  - — Regression Discontinuity (RD)
- `70_causal_inference/04_uplift.py`
  - — Uplift
  - — Persuadables / Sure-things / Lost-causes / Do-not-disturb
  - — S-learner (single model)
  - — T-learner (two models)
  - — X-learner (Künzel et al.)
  - — Qini curve
