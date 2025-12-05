from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df, df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2




# Дополнительные тесты для новых эвристик

# ---------------------------
# Тестовые DataFrame для разных эвристик
# ---------------------------

def df_high_unique_values():
    return pd.DataFrame({
        "unique_col": [f"id_{i}" for i in range(12)],  # 12 уникальных значений
        "dummy": [1]*12,
    })

def df_with_outliers():
    return pd.DataFrame({
        "num_col": [1]*11 + [100],  # последний элемент — выброс
    })

def df_mixed_types():
    return pd.DataFrame({
        "mixed": [1, 2, "three", 4, 5, 6, 7, 8, 9, 10, 11, 12],
    })

def df_imbalanced_cat():
    return pd.DataFrame({
        "cat": ["A"]*11 + ["B"],  # >90% одного значения
    })

def df_many_repeated_rows():
    return pd.DataFrame([
        [1, "A"],
        [1, "A"],  # повтор
        [1, "A"],
        [1, "A"],  # повтор
        [1, "A"],
        [1, "A"],  # повтор
        [4, "D"],
        [5, "E"],
        [6, "F"],
        [7, "G"],
        [8, "H"],
        [9, "I"],
    ], columns=["num", "cat"])

# ---------------------------
# Тесты
# ---------------------------

def test_high_ratio_of_unique_values():
    df = df_high_unique_values()
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)
    assert flags["has_high_ratio_of_unique_values"] is True

def test_has_outliers():
    df = df_with_outliers()
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)
    assert flags["has_outliers"] is True

def test_has_mixed_types():
    df = df_mixed_types()
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)
    assert flags["has_mixed_types"] is True

def test_has_imbalanced_categoricals():
    df = df_imbalanced_cat()
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)
    assert flags["has_imbalanced_categoricals"] is True

def test_has_many_repeated_rows():
    df = df_many_repeated_rows()
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)
    assert bool(flags["has_many_repeated_rows"]) is True
