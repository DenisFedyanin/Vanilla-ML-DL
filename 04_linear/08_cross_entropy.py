"""Концепция 43: Многоклассовая cross-entropy.

CE = -sum_i y_i * log(p_i), где y - one-hot, p - предсказание softmax.
"""
import numpy as np

p = np.array([0.7, 0.2, 0.1])      # предсказание softmax
y = np.array([1, 0, 0])            # настоящий класс - первый
ce = -np.sum(y * np.log(p + 1e-12))
print("CE =", ce)

# Если предсказали правильный класс с высокой вероятностью - CE мала
p2 = np.array([0.99, 0.005, 0.005])
print("CE при уверенно правильном:", -np.sum(y * np.log(p2 + 1e-12)))
