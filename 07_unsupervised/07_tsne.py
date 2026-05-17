"""Концепция 70: t-SNE - метод визуализации многомерных данных в 2D.

Сохраняет 'локальные' расстояния: близкие точки остаются близкими,
далёкие - могут перемешаться. Хорошо для просмотра, не для дальнейшего ML.
"""
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)
emb = TSNE(n_components=2, perplexity=30, random_state=0,
           init="pca").fit_transform(X[:300])
print("Сжали с", X.shape[1], "до", emb.shape[1], "измерений")
print("Первые 5 точек:\n", emb[:5])
