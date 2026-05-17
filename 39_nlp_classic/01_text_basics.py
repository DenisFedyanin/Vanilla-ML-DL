"""Раздел 39 — Классическое NLP. Файл 1: основы обработки текста.

Концепция: токенизация по пробелам и regex.
Простейшая: text.split(). Регулярка [A-Za-zА-Яа-я]+ выбирает только слова.
Знаки препинания, числа — отдельные обработки.

Концепция: нормализация (lowercasing).
Приведение к нижнему регистру стандартизует словарь.
Иногда вредит (имена собственные, аббревиатуры) — задача-зависимо.

Концепция: stopwords.
Частые служебные слова (и, в, the, of) обычно убирают — мало информации.

Концепция: stemming (Porter).
Грубое отбрасывание суффиксов: "running" -> "run". Морфологии не учитывает.

Концепция: лемматизация.
Приведение к словарной форме с учётом части речи: "is" -> "be", "лучше" -> "хороший".
Требует словаря/морфоанализатора. Точнее stemming, медленнее.
"""
import re

text = "The cats are running quickly and the dogs were barking loudly."

# === Tokenize: split vs regex ===
print("=== Tokenization ===")
toks_split = text.split()
toks_regex = re.findall(r"[A-Za-z]+", text)
print(f"split:  {toks_split}")
print(f"regex:  {toks_regex}")

# === Lowercase ===
toks = [t.lower() for t in toks_regex]
print(f"\nlowercased: {toks}")

# === Stopwords ===
STOPWORDS = {"the", "are", "and", "is", "a", "an", "of", "to", "in", "on", "were"}
filt = [t for t in toks if t not in STOPWORDS]
print(f"\nбез стоп-слов: {filt}")

# === Простой Porter-like стеммер: подмножество правил ===
def porter_lite(w):
    # Шаг 1a: множественное -> ед.
    for suf, repl in [("sses", "ss"), ("ies", "i"), ("ss", "ss"), ("s", "")]:
        if w.endswith(suf):
            w = w[:-len(suf)] + repl
            break
    # Шаг 1b: -ing/-ed
    for suf in ("ing", "ed"):
        if w.endswith(suf) and len(w) - len(suf) > 2:
            w = w[:-len(suf)]
            break
    # Шаг 2: -ly
    if w.endswith("ly") and len(w) > 4:
        w = w[:-2]
    return w


stems = [porter_lite(t) for t in filt]
print(f"\nstems:  {stems}")

# === Лемма vs стем (вручную крошечный словарь) ===
LEMMA = {
    "running": "run", "ran": "run", "runs": "run",
    "barking": "bark", "barked": "bark",
    "cats": "cat", "dogs": "dog",
    "quickly": "quick", "loudly": "loud",
}
lemmas = [LEMMA.get(t, t) for t in filt]
print(f"lemmas: {lemmas}")
print("\nЛемма даёт словарную форму (точно), стем — грубое усечение.")
