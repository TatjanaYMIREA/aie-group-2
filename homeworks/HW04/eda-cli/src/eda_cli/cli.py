from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .core import (
    DatasetSummary,
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)
from .viz import (
    plot_correlation_heatmap,
    plot_missing_matrix,
    plot_histograms_per_column,
    plot_histograms_with_outliers,
    plot_top_categories_barplots,
    save_top_categories_tables,
)

app = typer.Typer(help="Мини-CLI для EDA CSV-файлов")


def _load_csv(
    path: Path,
    sep: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"Файл '{path}' не найден")
    try:
        return pd.read_csv(path, sep=sep, encoding=encoding)
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter(f"Не удалось прочитать CSV: {exc}") from exc


@app.command()
def overview(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
) -> None:
    """
    Напечатать краткий обзор датасета:
    - размеры;
    - типы;
    - простая табличка по колонкам.
    """
    df = _load_csv(Path(path), sep=sep, encoding=encoding)
    summary: DatasetSummary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)

    typer.echo(f"Строк: {summary.n_rows}")
    typer.echo(f"Столбцов: {summary.n_cols}")
    typer.echo("\nКолонки:")
    typer.echo(summary_df.to_string(index=False))


@app.command()
def report(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    out_dir: str = typer.Option("reports", help="Каталог для отчёта."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
    max_hist_columns: int = typer.Option(6, help="Максимум числовых колонок для гистограмм."),
    top_k_categories: int = typer.Option(5, help="Сколько top-значений выводить для категориальных признаков."),
    title: str = typer.Option("EDA-отчёт", help="Заголовок отчёта."),
    min_missing_share: float = typer.Option(0.1, help="Порог доли пропусков, выше которого колонка считается проблемной."),
) -> None:
    """
    Сгенерировать полный EDA-отчёт:
    - текстовый overview и summary по колонкам (CSV/Markdown);
    - статистика пропусков;
    - корреляционная матрица;
    - top-k категорий по категориальным признакам;
    - картинки: гистограммы, матрица пропусков, heatmap корреляции.
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = _load_csv(Path(path), sep=sep, encoding=encoding)

    # 1. Обзор
    summary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)
    missing_df = missing_table(df)
    corr_df = correlation_matrix(df)
    top_cats = top_categories(df, top_k=top_k_categories)

    # 2. Качество в целом
    quality_flags = compute_quality_flags(summary, missing_df, df)

    # 3. Сохраняем табличные артефакты
    summary_df.to_csv(out_root / "summary.csv", index=False)
    if not missing_df.empty:
        missing_df.to_csv(out_root / "missing.csv", index=True)
    if not corr_df.empty:
        corr_df.to_csv(out_root / "correlation.csv", index=True)
    save_top_categories_tables(top_cats, out_root / "top_categories")

    # 4. Markdown-отчёт
    md_path = out_root / "report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"Исходный файл: `{Path(path).name}`\n\n")
        f.write(f"Строк: **{summary.n_rows}**, столбцов: **{summary.n_cols}**\n\n")

        f.write("## Параметры отчёта\n\n")
        f.write(f"- max_hist_columns: {max_hist_columns}\n")
        f.write(f"- top_k_categories: {top_k_categories}\n")
        f.write(f"- min_missing_share: {min_missing_share:.0%}\n\n")

        f.write("## Качество данных (эвристики)\n\n")
        f.write(f"- Слишком мало строк: **{quality_flags['too_few_rows']}**\n")
        f.write(f"- Слишком много колонок: **{quality_flags['too_many_columns']}**\n")
        f.write(f"- Макс. доля пропусков по колонке: **{quality_flags['max_missing_share']:.2%}**\n")
        f.write(f"- Слишком много пропусков: **{quality_flags['too_many_missing']}**\n")
        f.write(f"- Оценка качества: **{quality_flags['quality_score']:.2f}**\n\n")

        f.write("### Новые творческие эвристики\n\n")
        f.write(f"- new_высокий_процент_уникальных_значений (>80% уникальных значений): **{quality_flags['has_high_ratio_of_unique_values']}**\n")
        f.write(f"- new_выбросы (1.5*IQR, >5% значений): **{quality_flags['has_outliers']}**\n")
        f.write(f"- new_смешанные_типы (колонки с разными типами значений): **{quality_flags['has_mixed_types']}**\n")
        f.write(f"- new_несбалансированные_категориальные (>90% одного значения): **{quality_flags['has_imbalanced_categoricals']}**\n")
        f.write(f"- new_много_повторяющихся_строк (>10% повторов): **{quality_flags['has_many_repeated_rows']}**\n\n")

        f.write(f"## Колонки с пропусками выше {min_missing_share:.0%}\n\n")
        problematic_cols = missing_df[missing_df["missing_share"] > min_missing_share]
        if problematic_cols.empty:
            f.write("Нет колонок с пропусками выше порога.\n\n")
        else:
            f.write(problematic_cols.to_markdown() + "\n\n")

        f.write("## Колонки\n\n")
        f.write("См. файл `summary.csv`.\n\n")

        f.write("## Пропуски\n\n")
        if missing_df.empty:
            f.write("Пропусков нет или датасет пуст.\n\n")
        else:
            f.write("См. файлы `missing.csv` и `missing_matrix.png`.\n\n")

        f.write("## Корреляция числовых признаков\n\n")
        if corr_df.empty:
            f.write("Недостаточно числовых колонок для корреляции.\n\n")
        else:
            f.write("См. `correlation.csv` и `correlation_heatmap.png`.\n\n")

        f.write("## Категориальные признаки\n\n")
        if not top_cats:
            f.write("Категориальные/строковые признаки не найдены.\n\n")
        else:
            f.write(f"Топ-{top_k_categories} значений по колонкам см. в папке `top_categories/`\n\n")

        f.write("## Гистограммы числовых колонок\n\n")
        f.write("См. файлы `hist_*.png`.\n")

    # 5. Картинки
    plot_histograms_per_column(df, out_root, max_columns=max_hist_columns)
    plot_missing_matrix(df, out_root / "missing_matrix.png")
    plot_correlation_heatmap(df, out_root / "correlation_heatmap.png")

    # числовые гистограммы с выделением выбросов
    hist_outliers_dir = out_root / "histograms_outliers"
    plot_histograms_with_outliers(df, hist_outliers_dir, max_columns=max_hist_columns)

    # категориальные топ-k
    barplots_dir = out_root / "barplots_top_categories"
    plot_top_categories_barplots(df, barplots_dir, max_columns=5, top_k=top_k_categories)

    typer.echo(f"Отчёт сгенерирован в каталоге: {out_root}")
    typer.echo(f"- Основной markdown: {md_path}")
    typer.echo("- Табличные файлы: summary.csv, missing.csv, correlation.csv, top_categories/*.csv")
    typer.echo("- Графики: hist_*.png, missing_matrix.png, correlation_heatmap.png")
    typer.echo("### Новые творческие эвристики:")
    typer.echo(f"- new_высокий_процент_уникальных_значений: {quality_flags['has_high_ratio_of_unique_values']}")
    typer.echo(f"- new_выбросы: {quality_flags['has_outliers']}")
    typer.echo(f"- new_смешанные_типы: {quality_flags['has_mixed_types']}")
    typer.echo(f"- new_несбалансированные_категориальные: {quality_flags['has_imbalanced_categoricals']}")
    typer.echo(f"- new_много_повторяющихся_строк: {quality_flags['has_many_repeated_rows']}")


if __name__ == "__main__":
    app()
