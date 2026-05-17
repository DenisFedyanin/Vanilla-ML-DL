"""Концепции 73-77: Функции активации - ReLU, Sigmoid, Tanh, LeakyReLU, Softmax.

Активация делает сеть НЕлинейной. Без активаций - какое количество слоёв
ни ставь, всё равно получится одна линейная функция.
"""
import numpy as np


def relu(x):       return np.maximum(0, x)
def leaky(x, a=0.01): return np.where(x > 0, x, a * x)
def sigmoid(x):    return 1 / (1 + np.exp(-x))
def tanh(x):       return np.tanh(x)
def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()


z = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
print("input :", z)
print("relu  :", relu(z))
print("leaky :", leaky(z))
print("sigm  :", np.round(sigmoid(z), 3))
print("tanh  :", np.round(tanh(z), 3))
print("softmax:", np.round(softmax(z), 3))
