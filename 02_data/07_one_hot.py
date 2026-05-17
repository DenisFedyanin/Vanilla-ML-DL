"""Концепция 19: One-Hot encoding.

Категорию ["кошка","собака","рыба"] превращаем в столбцы [1,0,0], [0,1,0], [0,0,1].
"""
import numpy as np

labels = np.array(["кошка", "собака", "рыба", "кошка", "рыба"])
classes = np.unique(labels)
mapping = {c: i for i, c in enumerate(classes)}

onehot = np.zeros((len(labels), len(classes)), dtype=int)
for i, l in enumerate(labels):
    onehot[i, mapping[l]] = 1

print("Классы:", classes)
print("One-hot:\n", onehot)
