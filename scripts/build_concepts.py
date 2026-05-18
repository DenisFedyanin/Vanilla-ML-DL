"""Регенерирует CONCEPTS_2.md из docstring каждого файла глав.

Запуск из корня репо:
    python scripts/build_concepts.py

Логика: для каждого файла chapter/file.py берём первый docstring,
ищем строки вида `Концепция [NNN]: Заголовок.` и выводим их в Markdown.
"""
import re
import ast
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
CONCEPT_RE = re.compile(r'^Концепци[яи](?:\s+(\d+))?\s*:\s*(.+?)\.?\s*$')

CHAPTER_TITLES = {
    "01_basics": "Математика и numpy",
    "02_data": "Подготовка данных",
    "03_metrics": "Метрики качества",
    "04_linear": "Линейные модели",
    "05_regularization": "Регуляризация",
    "06_classic": "Классические алгоритмы",
    "07_unsupervised": "Обучение без учителя",
    "08_nn_basics": "Основы нейросетей",
    "09_architectures": "Архитектуры нейросетей",
    "10_optim_training": "Обучение и оптимизация",
    "11_extras": "Дополнительные концепции",
    "12_probability_distributions": "Распределения вероятностей",
    "13_statistics_tests": "Статистические тесты",
    "14_distance_metrics": "Метрики расстояний",
    "15_feature_engineering": "Feature engineering",
    "16_preprocessing": "Препроцессинг",
    "17_linear_models_extended": "Расширенные линейные модели",
    "18_svm_family": "Семейство SVM",
    "19_trees_extended": "Расширенные деревья",
    "20_boosting_family": "Бустинг",
    "21_clustering_extended": "Расширенная кластеризация",
    "22_dim_reduction_extended": "Понижение размерности",
    "23_matrix_factorization": "Матричная факторизация",
    "24_associations": "Ассоциативные правила",
    "25_anomaly_detection": "Поиск аномалий",
    "26_semi_supervised": "Semi-supervised обучение",
    "27_self_supervised": "Self-supervised обучение",
    "28_neural_layers": "Слои нейросетей",
    "29_activations_extended": "Функции активации",
    "30_optimizers_extended": "Оптимизаторы",
    "31_losses_extended": "Функции потерь",
    "32_regularization_extended": "Расширенная регуляризация",
    "33_init_extended": "Инициализация весов",
    "34_normalization": "Нормализация",
    "35_cnn_archs": "Архитектуры CNN",
    "36_rnn_extended": "Расширенные RNN",
    "37_attention": "Механизмы attention",
    "38_transformers": "Архитектура Transformer",
    "39_nlp_classic": "Классический NLP",
    "40_nlp_embeddings": "Word embeddings",
    "41_lm_pretrained": "Предобученные языковые модели",
    "42_cv_basics": "Основы computer vision",
    "43_cv_classic": "Классический computer vision",
    "44_cv_detection": "Детекция объектов",
    "45_cv_segmentation": "Сегментация изображений",
    "46_generative_classic": "Классические генеративные модели",
    "47_generative_modern": "Современные генеративные модели",
    "48_rl_extended": "Reinforcement learning",
    "49_mlops_basics": "Основы MLOps",
    "50_interpretability": "Интерпретируемость моделей",
    "51_graph_ml": "Graph ML",
    "52_time_series": "Временные ряды",
    "53_recommender": "Рекомендательные системы",
    "54_bayesian_ml": "Байесовский ML",
    "55_mcmc": "MCMC и вариационный вывод",
    "56_optimization_extended": "Расширенная оптимизация",
    "57_hpo_extended": "Подбор гиперпараметров",
    "58_info_theory": "Теория информации",
    "59_linalg_extended": "Расширенная линейная алгебра",
    "60_calculus_extended": "Расширенный матанализ",
    "61_probability_extended": "Расширенная теория вероятностей",
    "62_tricks_training": "Трюки обучения",
    "63_data_augmentation": "Аугментация данных",
    "64_meta_learning": "Meta-learning",
    "65_metric_learning": "Metric learning",
    "66_active_learning": "Active learning",
    "67_online_learning": "Online learning",
    "68_kernel_methods": "Kernel methods",
    "69_density_estimation": "Density estimation",
    "70_causal_inference": "Causal inference",
}


def first_doc(path):
    try:
        return ast.get_docstring(ast.parse(path.read_text(encoding='utf-8'))) or ''
    except SyntaxError:
        return ''


def extract(doc):
    return [(m.group(1), m.group(2).strip())
            for m in (CONCEPT_RE.match(l.strip()) for l in doc.splitlines())
            if m]


def main():
    files = sorted(
        p for p in ROOT.rglob('*.py')
        if re.match(r'^\d+_', p.relative_to(ROOT).parts[0])
        and p.name != 'run_all.py'
        and 'scripts' not in p.parts
    )
    by_chapter = defaultdict(list)
    total = 0
    for f in files:
        items = extract(first_doc(f))
        total += len(items)
        by_chapter[f.relative_to(ROOT).parts[0]].append(
            (f.relative_to(ROOT).as_posix(), items)
        )

    out = []
    out.append("# CONCEPTS — карта концепций по файлам\n")
    out.append("Автогенерируется из docstring каждого файла глав.")
    out.append(f"Всего {len(files)} файлов, {total} концепций в {len(by_chapter)} разделах.\n")
    out.append("Регенерация: `python scripts/build_concepts.py`.\n")

    out.append("## Оглавление\n")
    for ch in sorted(by_chapter):
        n = sum(len(items) for _, items in by_chapter[ch])
        title = CHAPTER_TITLES.get(ch, ch)
        anchor = ch.lower().replace('_', '-')
        out.append(f"- [{ch} — {title}](#{anchor}) ({n} концепций)")

    for ch in sorted(by_chapter):
        title = CHAPTER_TITLES.get(ch, ch)
        out.append(f"\n## {ch} — {title}\n")
        for path, items in by_chapter[ch]:
            if not items:
                out.append(f"- `{path}`  *(нет извлекаемых docstring-концепций)*")
                continue
            out.append(f"- `{path}`")
            for num, ttl in items:
                prefix = f"**{num}.** " if num else "— "
                out.append(f"  - {prefix}{ttl}")

    (ROOT / "CONCEPTS_2.md").write_text("\n".join(out) + "\n", encoding='utf-8')
    print(f"CONCEPTS_2.md: {len(files)} files, {total} concepts, {len(by_chapter)} chapters")


if __name__ == "__main__":
    main()
