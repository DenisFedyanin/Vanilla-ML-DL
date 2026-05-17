"""Концепция 118: Генетический алгоритм.

'Особи' - кандидаты решения. Лучшие 'размножаются' (crossover) и
'мутируют'. Эволюционная оптимизация.
Задача: подобрать строку 'HELLO_WORLD'.
"""
import random

TARGET = "HELLO_WORLD"
ALPH = "ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
POP, MUT = 100, 0.05
random.seed(0)


def fitness(s): return sum(a == b for a, b in zip(s, TARGET))


def make(): return "".join(random.choices(ALPH, k=len(TARGET)))


def mutate(s):
    return "".join(c if random.random() > MUT else random.choice(ALPH) for c in s)


def crossover(a, b):
    cut = random.randint(1, len(TARGET) - 1)
    return a[:cut] + b[cut:]


pop = [make() for _ in range(POP)]
for gen in range(500):
    pop.sort(key=fitness, reverse=True)
    if pop[0] == TARGET:
        print(f"Найдено за {gen} поколений: {pop[0]}"); break
    parents = pop[:20]
    pop = parents + [mutate(crossover(random.choice(parents),
                                      random.choice(parents)))
                     for _ in range(POP - len(parents))]
else:
    print(f"Лучший: {pop[0]} (score={fitness(pop[0])}/{len(TARGET)})")
