# Vanilla-ML-DL — 100+ концепций машинного обучения «на пальцах»

Учебный проект для тех, кто только начинает (даже для школьника).
Внутри — **больше 100 концепций ML и Deep Learning**, разобранных
на коротких самостоятельных скриптах на чистом Python + numpy.
Никакой магии: всё, что можно — реализовано вручную, без «чёрных ящиков».

## Что нужно

* Python 3.9 или новее
* 3 библиотеки: `numpy`, `matplotlib`, `scikit-learn`

Установка:

```bash
pip install -r requirements.txt
```

## Как запускать

Каждый файл можно запустить отдельно, например:

```bash
python 01_basics/01_vectors.py
python 08_nn_basics/07_backprop_manual.py
```

Или запустить меню со списком всех демо:

```bash
python run_all.py
```

## Структура (11 разделов, 115+ концепций)

1. `01_basics/` — векторы, матрицы, нормы, градиенты, базовая статистика
2. `02_data/` — подготовка данных, train/test, нормализация, кодирование
3. `03_metrics/` — метрики качества (MAE, F1, ROC-AUC и т.д.)
4. `04_linear/` — линейная и логистическая регрессия, GD, SGD, softmax
5. `05_regularization/` — L1, L2, Elastic Net, early stopping, dropout
6. `06_classic/` — KNN, Naive Bayes, деревья, бэггинг, бустинг, SVM, перцептрон
7. `07_unsupervised/` — k-means, иерархическая, DBSCAN, PCA, SVD, t-SNE
8. `08_nn_basics/` — нейрон, активации, forward/backward, инициализация, Adam
9. `09_architectures/` — MLP, свёртка, pooling, CNN, RNN, LSTM, autoencoder, embeddings
10. `10_optim_training/` — bias/variance, overfitting, learning curves, grid search
11. `11_extras/` — байес, MLE, KL-дивергенция, Q-learning, генетика, bootstrap

## Полный список концепций

См. файл [`CONCEPTS.md`](CONCEPTS.md).

## Главное правило

Открой любой файл, прочитай комментарии, запусти, поменяй параметры,
посмотри, что изменилось. Это и есть обучение.
