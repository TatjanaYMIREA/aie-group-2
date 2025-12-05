from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PathLike = Union[str, Path]


def _ensure_dir(path: PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_histograms_per_column(
    df: pd.DataFrame,
    out_dir: PathLike,
    max_columns: int = 6,
    bins: int = 20,
) -> List[Path]:
    """
    Для числовых колонок строит по отдельной гистограмме.
    Возвращает список путей к PNG.
    """
    out_dir = _ensure_dir(out_dir)
    numeric_df = df.select_dtypes(include="number")

    paths: List[Path] = []
    for i, name in enumerate(numeric_df.columns[:max_columns]):
        s = numeric_df[name].dropna()
        if s.empty:
            continue

        fig, ax = plt.subplots()
        ax.hist(s.values, bins=bins)
        ax.set_title(f"Histogram of {name}")
        ax.set_xlabel(name)
        ax.set_ylabel("Count")
        fig.tight_layout()

        out_path = out_dir / f"hist_{i+1}_{name}.png"
        fig.savefig(out_path)
        plt.close(fig)

        paths.append(out_path)

    return paths


def plot_missing_matrix(df: pd.DataFrame, out_path: PathLike) -> Path:
    """
    Простая визуализация пропусков: где True=пропуск, False=значение.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        # Рисуем пустой график
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Empty dataset", ha="center", va="center")
        ax.axis("off")
    else:
        mask = df.isna().values
        fig, ax = plt.subplots(figsize=(min(12, df.shape[1] * 0.4), 4))
        ax.imshow(mask, aspect="auto", interpolation="none")
        ax.set_xlabel("Columns")
        ax.set_ylabel("Rows")
        ax.set_title("Missing values matrix")
        ax.set_xticks(range(df.shape[1]))
        ax.set_xticklabels(df.columns, rotation=90, fontsize=8)
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_correlation_heatmap(df: pd.DataFrame, out_path: PathLike) -> Path:
    """
    Тепловая карта корреляции числовых признаков.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Not enough numeric columns for correlation", ha="center", va="center")
        ax.axis("off")
    else:
        corr = numeric_df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(min(10, corr.shape[1]), min(8, corr.shape[0])))
        im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
        ax.set_xticks(range(corr.shape[1]))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
        ax.set_yticks(range(corr.shape[0]))
        ax.set_yticklabels(corr.index, fontsize=8)
        ax.set_title("Correlation heatmap")
        fig.colorbar(im, ax=ax, label="Pearson r")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def save_top_categories_tables(
    top_cats: Dict[str, pd.DataFrame],
    out_dir: PathLike,
) -> List[Path]:
    """
    Сохраняет top-k категорий по колонкам в отдельные CSV.
    """
    out_dir = _ensure_dir(out_dir)
    paths: List[Path] = []
    for name, table in top_cats.items():
        out_path = out_dir / f"top_values_{name}.csv"
        table.to_csv(out_path, index=False)
        paths.append(out_path)
    return paths

def plot_top_categories_barplots(
    df: pd.DataFrame,
    out_dir: PathLike,
    max_columns: int = 5,
    top_k: int = 5,
) -> List[Path]:
    """
    Для категориальных колонок строит bar plot топ-k значений.
    Имена файлов включают колонку и top_k: category_<col>_top<k>.png
    """
    out_dir = _ensure_dir(out_dir)
    cat_cols = [c for c in df.columns if df[c].dtype == "object" or isinstance(df[c].dtype, pd.CategoricalDtype)]
    
    paths: List[Path] = []
    for i, col in enumerate(cat_cols[:max_columns]):
        s = df[col].dropna()
        if s.empty:
            continue
        top_values = s.value_counts().head(top_k)
        fig, ax = plt.subplots()
        ax.bar(top_values.index.astype(str), top_values.values)
        ax.set_title(f"Top-{top_k} categories of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.tick_params(axis='x', rotation=45)
        fig.tight_layout()

        # Новое имя файла с включением top_k
        out_path = Path(out_dir) / f"category_{col}_top{top_k}.png"
        fig.savefig(out_path)
        plt.close(fig)
        paths.append(out_path)
    return paths


def plot_histograms_with_outliers(df: pd.DataFrame, out_dir: PathLike, bins: int = 20, max_columns: int = 6):
    out_dir = _ensure_dir(out_dir)
    numeric_df = df.select_dtypes(include="number")
    paths = []
    
    for i, col in enumerate(numeric_df.columns[:max_columns]):
        s = numeric_df[col].dropna()
        if s.empty:
            continue
        
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        fig, ax = plt.subplots()
        ax.hist(s[(s >= lower) & (s <= upper)], bins=bins, color='skyblue', label='main')
        ax.hist(s[(s < lower) | (s > upper)], bins=bins, color='red', label='outliers')
        ax.set_title(f"Histogram with outliers: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.legend()
        fig.tight_layout()
        
        out_path = Path(out_dir) / f"hist_outliers_{i+1}_{col}.png"
        fig.savefig(out_path)
        plt.close(fig)
        paths.append(out_path)
    return paths
