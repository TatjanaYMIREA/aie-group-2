from __future__ import annotations

from time import perf_counter
from typing import Dict

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .core import compute_quality_flags, missing_table, summarize_dataset

app = FastAPI(
    title="AIE Dataset Quality API",
    version="0.3.1",
    description="HTTP-сервис для оценки качества датасетов с новыми флагами HW03.",
    docs_url="/docs",
    redoc_url=None,
)


class QualityRequest(BaseModel):
    n_rows: int = Field(..., ge=0)
    n_cols: int = Field(..., ge=0)
    max_missing_share: float = Field(..., ge=0.0, le=1.0)
    numeric_cols: int = Field(..., ge=0)
    categorical_cols: int = Field(..., ge=0)


class QualityResponse(BaseModel):
    ok_for_model: bool
    quality_score: float = Field(..., ge=0.0, le=1.0)
    message: str
    latency_ms: float
    flags: dict[str, bool] | None = None
    dataset_shape: dict[str, int] | None = None


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok", "service": "dataset-quality", "version": "0.3.1"}


# ---------- Заглушка /quality ----------

@app.post("/quality", response_model=QualityResponse, tags=["quality"])
def quality(req: QualityRequest) -> QualityResponse:
    start = perf_counter()

    # Старое вычисление качества
    score = 1.0
    score -= req.max_missing_share
    if req.n_rows < 1000:
        score -= 0.2
    if req.n_cols > 100:
        score -= 0.1
    if req.numeric_cols == 0 and req.categorical_cols > 0:
        score -= 0.1
    if req.categorical_cols == 0 and req.numeric_cols > 0:
        score -= 0.05
    score = max(0.0, min(1.0, score))
    ok_for_model = score >= 0.7
    message = (
        "Данных достаточно, модель можно обучать (по текущим эвристикам)."
        if ok_for_model
        else "Качество данных недостаточно, требуется доработка (по текущим эвристикам)."
    )

    latency_ms = (perf_counter() - start) * 1000.0

    # Старые флаги (без новых эвристик)
    flags: dict[str, bool] = {
        "too_few_rows": req.n_rows < 1000,
        "too_many_columns": req.n_cols > 100,
        "max_missing_share": req.max_missing_share > 0,
        "too_many_missing": req.max_missing_share > 0.5,
    }

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags,
        dataset_shape={"n_rows": req.n_rows, "n_cols": req.n_cols},
    )


# ---------- /quality-from-csv (старые флаги, старое качество) ----------

@app.post("/quality-from-csv", response_model=QualityResponse, tags=["quality"])
async def quality_from_csv(file: UploadFile = File(...)) -> QualityResponse:
    start = perf_counter()

    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Ожидается CSV-файл (content-type text/csv).")

    try:
        df = pd.read_csv(file.file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV-файл не содержит данных (пустой DataFrame).")

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags_all = compute_quality_flags(summary, missing_df, df)

    # Старые флаги
    flags_old_only: dict[str, bool] = {
        "too_few_rows": bool(flags_all.get("too_few_rows", False)),
        "too_many_columns": bool(flags_all.get("too_many_columns", False)),
        "max_missing_share": bool(flags_all.get("max_missing_share", False)),
        "too_many_missing": bool(flags_all.get("too_many_missing", False)),
    }

    # Старое качество (без новых эвристик)
    score = 1.0
    max_missing_share = float(missing_df["missing_share"].max()) if not missing_df.empty else 0.0
    score -= max_missing_share
    if flags_old_only["too_few_rows"]:
        score -= 0.2
    if flags_old_only["too_many_columns"]:
        score -= 0.1
    if flags_old_only["too_many_missing"]:
        score -= 0.1
    score = max(0.0, min(1.0, score))
    ok_for_model = score >= 0.7
    message = (
        "CSV достаточно качественный для модели."
        if ok_for_model
        else "CSV требует доработки перед обучением модели."
    )

    latency_ms = (perf_counter() - start) * 1000.0
    n_rows, n_cols = int(df.shape[0]), int(df.shape[1])

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags_old_only,
        dataset_shape={"n_rows": n_rows, "n_cols": n_cols},
    )


# ---------- /quality-flags-from-csv (все флаги, новое качество) ----------

@app.post("/quality-flags-from-csv", response_model=QualityResponse, tags=["quality"])
async def quality_flags_from_csv(file: UploadFile = File(...)) -> QualityResponse:
    start = perf_counter()

    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Ожидается CSV-файл (content-type text/csv).")

    try:
        df = pd.read_csv(file.file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV-файл не содержит данных (пустой DataFrame).")

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags_all = compute_quality_flags(summary, missing_df, df)

    # Для нового эндпоинта используем новое качество из flags['quality_score']
    score = float(flags_all.get("quality_score", 0.0))
    score = max(0.0, min(1.0, score))
    ok_for_model = score >= 0.7
    message = (
        "CSV достаточно качественный для модели."
        if ok_for_model
        else "CSV требует доработки перед обучением модели."
    )
    # Все флаги
    flags_bool = {k: bool(v) for k, v in flags_all.items() if k != "quality_score"}

    latency_ms = (perf_counter() - start) * 1000.0
    n_rows, n_cols = int(df.shape[0]), int(df.shape[1])

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags_bool,
        dataset_shape={"n_rows": n_rows, "n_cols": n_cols},
    )
