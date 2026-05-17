"""Простое меню запуска всех демо.

Запуск:
    python run_all.py

Можно выбрать раздел и пример, либо запустить все подряд.
"""
import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent


def list_demos():
    demos = []
    for chapter in sorted(ROOT.iterdir()):
        if not chapter.is_dir() or not chapter.name[0].isdigit():
            continue
        scripts = sorted(p for p in chapter.iterdir() if p.suffix == ".py")
        demos.append((chapter.name, scripts))
    return demos


def main():
    demos = list_demos()
    print("=" * 60)
    print("Vanilla-ML-DL: 100+ концепций ML/DL")
    print("=" * 60)
    for ch_name, scripts in demos:
        print(f"\n[{ch_name}] ({len(scripts)} демо)")
        for s in scripts:
            print(f"   {s.relative_to(ROOT)}")
    print("\n" + "=" * 60)
    print("Введите путь к скрипту (или 'all' чтобы запустить все, 'q' — выход):")
    choice = input("> ").strip()
    if choice in ("q", "Q", ""):
        return
    if choice == "all":
        for _, scripts in demos:
            for s in scripts:
                print(f"\n--- {s.relative_to(ROOT)} ---")
                subprocess.run([sys.executable, str(s)])
        return
    path = ROOT / choice
    if path.exists():
        subprocess.run([sys.executable, str(path)])
    else:
        print("Не найден.")


if __name__ == "__main__":
    main()
