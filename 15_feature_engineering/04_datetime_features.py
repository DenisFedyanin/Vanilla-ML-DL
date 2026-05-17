"""Раздел 15 — Feature Engineering.

Файл 4: признаки из дат/времени.

Концепция 220: Календарные признаки.
Из datetime извлекают year, month, day, hour, minute, weekday. Простые
интервальные/циклические признаки для моделей.

Концепция 221: Циклическое sin/cos-кодирование.
Чтобы модель понимала, что 23 час близок к 0 часу:
hour_sin = sin(2*pi*hour/24), hour_cos = cos(2*pi*hour/24).

Концепция 222: Флаг рабочего дня.
weekday < 5 -> 1 (Пн..Пт), иначе 0. Часто резко меняет поведение метрик.

Концепция 223: Days-since-event.
Разница между текущей датой и опорной (например, регистрация пользователя).
Превращает дату в непрерывный признак.

Концепция 224: Простой holiday-флаг.
Совпадает ли дата со списком праздников. Здесь — заранее заданный список.

Концепция 225: Сезонность через циклы по дню года.
sin/cos(2*pi*day_of_year/365). Хорошо ловит температуру, продажи и т.п.
"""
from datetime import date, datetime

import numpy as np

ts = [
    datetime(2024, 1, 5, 10, 0),
    datetime(2024, 5, 1, 23, 30),
    datetime(2024, 8, 15, 0, 5),
    datetime(2024, 12, 31, 18, 45),
]

# 220: extract
years = np.array([t.year for t in ts])
months = np.array([t.month for t in ts])
days = np.array([t.day for t in ts])
hours = np.array([t.hour for t in ts])
weekdays = np.array([t.weekday() for t in ts])
print(f"220 year={years}, month={months}, day={days}, hour={hours}, wd={weekdays}")

# 221: cyclic
hour_sin = np.sin(2 * np.pi * hours / 24)
hour_cos = np.cos(2 * np.pi * hours / 24)
print(f"221 hour_sin={hour_sin.round(3)}, hour_cos={hour_cos.round(3)}")

# 222: business day
is_business = (weekdays < 5).astype(int)
print(f"222 is_business={is_business}")

# 223: days since reference
ref = date(2024, 1, 1)
dsr = np.array([(t.date() - ref).days for t in ts])
print(f"223 days_since_2024-01-01={dsr}")

# 224: holiday flag
holidays = {date(2024, 1, 1), date(2024, 5, 1), date(2024, 12, 31)}
is_holiday = np.array([1 if t.date() in holidays else 0 for t in ts])
print(f"224 is_holiday={is_holiday}")

# 225: seasonality (day of year cyclic)
doy = np.array([t.timetuple().tm_yday for t in ts])
season_sin = np.sin(2 * np.pi * doy / 365)
season_cos = np.cos(2 * np.pi * doy / 365)
print(f"225 doy={doy}, season_sin={season_sin.round(3)}, season_cos={season_cos.round(3)}")
