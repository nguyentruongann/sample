"""Read-only HTTP API for the latest scheduled demand forecasts."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import h3

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .basemap import render_hcm_basemap
from .dashboard import load_dashboard_snapshot, load_hex_detail
from .database import (
    database_is_ready,
    ensure_runtime_tables,
    load_latest_predictions,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_runtime_tables(get_settings())
    yield


app = FastAPI(
    title="Demand Forecasting API",
    version="1.0.0",
    description="Latest H10/H30/H60 forecasts produced by the Airflow pipeline.",
    lifespan=lifespan,
)

ASSETS = Path(__file__).resolve().parent / "dashboard_assets"
app.mount("/dashboard/assets", StaticFiles(directory=ASSETS), name="dashboard-assets")


@app.get("/dashboard", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(
        ASSETS / "index.html",
        media_type="text/html",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/v1/dashboard/basemap", tags=["dashboard"], response_class=HTMLResponse)
def dashboard_basemap() -> HTMLResponse:
    return HTMLResponse(
        render_hcm_basemap(),
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/v1/dashboard/snapshot", tags=["dashboard"])
def dashboard_snapshot(
    travel_mode: str = Query(default="Car", pattern="^(Car|Motorcycle)$"),
    horizon_minutes: int = Query(default=10),
    forecast_start_utc: datetime | None = None,
) -> dict:
    if horizon_minutes not in (10, 30, 60):
        raise HTTPException(status_code=422, detail="Choose H10, H30 or H60")
    if forecast_start_utc is not None and forecast_start_utc.tzinfo is None:
        raise HTTPException(status_code=422, detail="Timestamp must include a timezone")
    result = load_dashboard_snapshot(
        get_settings(), travel_mode, horizon_minutes, forecast_start_utc
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Chưa có dự báo cho chế độ và thời điểm này.")
    return result


@app.get("/v1/dashboard/hex/{hex_id_7}", tags=["dashboard"])
def dashboard_hex(
    hex_id_7: str,
    forecast_start_utc: datetime,
    travel_mode: str = Query(default="Car", pattern="^(Car|Motorcycle)$"),
) -> dict:
    if not h3.is_valid_cell(hex_id_7) or h3.get_resolution(hex_id_7) != 7:
        raise HTTPException(status_code=422, detail="H3 resolution 7 cell required")
    if forecast_start_utc.tzinfo is None:
        raise HTTPException(status_code=422, detail="Timestamp must include a timezone")
    return load_hex_detail(get_settings(), travel_mode, hex_id_7, forecast_start_utc)


@app.get("/health", tags=["operations"])
def health() -> dict[str, str]:
    if not database_is_ready(get_settings()):
        raise HTTPException(status_code=503, detail="PostgreSQL is unavailable")
    return {"status": "ok"}


@app.get("/v1/predictions/latest", tags=["predictions"])
def latest_predictions(
    travel_mode: str | None = Query(default=None, pattern="^(Car|Motorcycle)$"),
    horizon_minutes: int | None = Query(default=None, ge=10, le=1440),
    hex_id_7: str | None = Query(default=None, min_length=1, max_length=32),
    limit: int = Query(default=1000, ge=1, le=10_000),
) -> dict:
    frame = load_latest_predictions(
        settings=get_settings(),
        travel_mode=travel_mode,
        horizon_minutes=horizon_minutes,
        hex_id_7=hex_id_7,
        limit=limit,
    )
    if frame.empty:
        raise HTTPException(
            status_code=404,
            detail="No predictions found. Train models and run the live inference DAG first.",
        )

    for column in ("forecast_start_utc", "created_at"):
        frame[column] = frame[column].astype(str)
    return {
        "forecast_start_utc": frame["forecast_start_utc"].iloc[0],
        "count": len(frame),
        "predictions": frame.to_dict(orient="records"),
    }
