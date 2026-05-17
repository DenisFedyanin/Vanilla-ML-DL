"""Раздел 26 — Semi-supervised. Файл 3: концепты продвинутых методов.

Концепция 698: Co-training.
Признаковое пространство делят на 2 «представления» (например, текст и
картинка). Учат две модели независимо, каждая помечает уверенно для другой.
Требует условной независимости представлений.

Концепция 699: Tri-training.
Три модели вместо двух. Объект получает псевдо-метку, если двое из трёх
согласны и уверены. Меньше требований к независимости представлений.

Концепция 700: Mean Teacher.
Студент и учитель — одна архитектура; веса учителя — EMA весов студента.
Студент учится на разметке + на согласовании с учителем на неразмеченных
(consistency loss). Очень популярен в SSL для изображений.

Концепция 701: Π-model.
Минимизирует расхождение предсказаний модели на двух разных аугментациях
одного и того же неразмеченного объекта. Простой и эффективный baseline.

Концепция 702: FixMatch.
Сильная и слабая аугментация. Слабая → предсказание (если уверенность > tau,
получаем псевдо-метку). Сильная аугментация того же объекта учится
предсказывать ту же псевдо-метку. Сочетает consistency и pseudo-labeling.
"""
import numpy as np

# Этот файл — концептуальный. Печатаем краткие пояснения и формальные шаги.
rng = np.random.default_rng(0)

steps = {
    698: ("Co-training", [
        "1. разделить признаки на V1, V2",
        "2. обучить f1 на (L, V1), f2 на (L, V2)",
        "3. f1 размечает уверенные U для f2 и наоборот",
        "4. повторять",
    ]),
    699: ("Tri-training", [
        "1. обучить f1, f2, f3 на bootstrap-выборках L",
        "2. для x in U: если f_i(x)=f_j(x) и уверены — псевдо-метка для f_k",
        "3. переобучить f_k на (L + новые псевдо-метки)",
    ]),
    700: ("Mean Teacher", [
        "1. teacher_weights = EMA(student_weights, alpha=0.99)",
        "2. supervised_loss = CE(student(x_L), y_L)",
        "3. consistency_loss = MSE(student(x_U + aug1), teacher(x_U + aug2))",
        "4. loss = sup + lambda(t) * consist; lambda растёт с эпохой",
    ]),
    701: ("Π-model", [
        "1. два прохода с разной аугментацией x -> y1, y2",
        "2. loss = sup(L) + w * ||y1 - y2||^2 (на U)",
        "3. w линейно увеличивается с эпохой (ramp-up)",
    ]),
    702: ("FixMatch", [
        "1. weak_aug(x_U) -> q; если max(q) > tau: pseudo = argmax(q)",
        "2. strong_aug(x_U) -> p; loss_U = CE(p, pseudo) только для уверенных",
        "3. loss = CE(L) + lambda * loss_U",
    ]),
}
for cid, (name, ops) in steps.items():
    print(f"{cid} {name}:")
    for line in ops:
        print(f"     {line}")

# Микро-демо EMA весов (Mean Teacher) на числах для иллюстрации.
student_w = 0.0
teacher_w = 0.0
alpha = 0.9
trajectory = []
for t in range(10):
    student_w = student_w + rng.normal()  # шаг SGD
    teacher_w = alpha * teacher_w + (1 - alpha) * student_w
    trajectory.append((round(student_w, 2), round(teacher_w, 2)))
print(f"700 EMA-демо (student, teacher) по итерациям: {trajectory}")
