"""Small, read-only queries for the live H3 presentation dashboard."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache

import h3
import psycopg
from psycopg import sql

from .config import Settings
from .database import _validate_identifier


def _utc(value: datetime | None) -> str | None:
    return value.astimezone(timezone.utc).isoformat() if value is not None else None


@lru_cache(maxsize=4096)
def _shape(hex_id: str) -> dict | None:
    # A bad ID in a historical import must not break the entire map.
    if not h3.is_valid_cell(hex_id) or h3.get_resolution(hex_id) != 7:
        return None
    latitude, longitude = h3.cell_to_latlng(hex_id)
    return {
        "center": [longitude, latitude],
        "boundary": [[lng, lat] for lat, lng in h3.cell_to_boundary(hex_id)],
    }


def _tables(settings: Settings) -> tuple[sql.Composed, sql.Composed]:
    schema = sql.Identifier(_validate_identifier(settings.postgres_schema, "schema"))
    raw = sql.SQL("{}.{}").format(
        schema, sql.Identifier(_validate_identifier(settings.postgres_table, "table"))
    )
    predictions = sql.SQL("{}.{}").format(
        schema, sql.Identifier(_validate_identifier(settings.predictions_table, "table"))
    )
    return raw, predictions


def load_dashboard_snapshot(
    settings: Settings,
    travel_mode: str,
    horizon_minutes: int,
    forecast_start_utc: datetime | None = None,
) -> dict | None:
    """Return one complete forecast layer and the newest observed layer.

    The selected batch is scoped by both mode and horizon, so a partial new
    inference run never makes another mode appear empty.
    """

    raw, predictions = _tables(settings)
    with psycopg.connect(settings.require_database_url(), connect_timeout=5) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            if forecast_start_utc is None:
                cursor.execute(
                    sql.SQL(
                        "SELECT forecast_start_utc FROM {} "
                        "WHERE travel_mode = %s AND horizon_minutes = %s "
                        "AND model_name = 'lightgbm' ORDER BY forecast_start_utc DESC LIMIT 1"
                    ).format(predictions),
                    (travel_mode, horizon_minutes),
                )
                row = cursor.fetchone()
                if row is None:
                    return None
                forecast_start_utc = row[0]

            cursor.execute(
                sql.SQL(
                    "SELECT hex_id_7, predicted_demand, created_at, model_version "
                    "FROM {} WHERE forecast_start_utc = %s AND travel_mode = %s "
                    "AND horizon_minutes = %s AND model_name = 'lightgbm'"
                ).format(predictions),
                (forecast_start_utc, travel_mode, horizon_minutes),
            )
            forecast_rows = cursor.fetchall()
            if not forecast_rows:
                return None

            cursor.execute(
                sql.SQL(
                    "SELECT forecast_start_utc FROM {} WHERE travel_mode = %s "
                    "AND horizon_minutes = %s AND model_name = 'lightgbm' "
                    "GROUP BY forecast_start_utc ORDER BY forecast_start_utc DESC LIMIT 18"
                ).format(predictions),
                (travel_mode, horizon_minutes),
            )
            available_times = [_utc(row[0]) for row in cursor.fetchall()]

            cursor.execute(sql.SQL("SELECT MAX(period_datetime_utc) FROM {}").format(raw))
            observed_at = cursor.fetchone()[0]
            observed_rows: list[tuple] = []
            region_history: list[dict] = []
            if observed_at is not None:
                cursor.execute(
                    sql.SQL(
                        "SELECT hex_id_7, total_demand FROM {} "
                        "WHERE period_datetime_utc = %s AND travel_mode = %s"
                    ).format(raw),
                    (observed_at, travel_mode),
                )
                observed_rows = cursor.fetchall()
                cursor.execute(
                    sql.SQL(
                        "SELECT period_datetime_utc, SUM(total_demand) FROM {} "
                        "WHERE period_datetime_utc > %s - INTERVAL '2 hours' "
                        "AND period_datetime_utc <= %s AND travel_mode = %s "
                        "GROUP BY period_datetime_utc ORDER BY period_datetime_utc"
                    ).format(raw),
                    (observed_at, observed_at, travel_mode),
                )
                region_history = [
                    {"at": _utc(at), "demand": int(total)} for at, total in cursor.fetchall()
                ]

    observations = {hex_id: int(value) for hex_id, value in observed_rows}
    cells = []
    unmapped = 0
    for hex_id, value, _, _ in forecast_rows:
        shape = _shape(hex_id)
        if shape is None:
            unmapped += 1
            continue
        cells.append(
            {
                "hex_id_7": hex_id,
                "predicted_demand": round(float(value), 2),
                "observed_demand": observations.get(hex_id),
                **shape,
            }
        )

    return {
        "travel_mode": travel_mode,
        "horizon_minutes": horizon_minutes,
        "forecast_start_utc": _utc(forecast_start_utc),
        "generated_at_utc": _utc(max(row[2] for row in forecast_rows)),
        "observed_at_utc": _utc(observed_at),
        "model_version": forecast_rows[0][3],
        "available_times": available_times,
        "region_history": region_history,
        "observed_total": sum(observations.values()),
        "cells": cells,
        "unmapped_cells": unmapped,
        "data_kind": "synthetic",
    }


def load_hex_detail(
    settings: Settings,
    travel_mode: str,
    hex_id_7: str,
    forecast_start_utc: datetime,
) -> dict:
    raw, predictions = _tables(settings)
    with psycopg.connect(settings.require_database_url(), connect_timeout=5) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            cursor.execute(sql.SQL("SELECT MAX(period_datetime_utc) FROM {}").format(raw))
            observed_at = cursor.fetchone()[0]
            history = []
            if observed_at is not None:
                cursor.execute(
                    sql.SQL(
                        "SELECT period_datetime_utc, total_demand FROM {} "
                        "WHERE travel_mode = %s AND hex_id_7 = %s "
                        "AND period_datetime_utc > %s - INTERVAL '2 hours' "
                        "AND period_datetime_utc <= %s ORDER BY period_datetime_utc"
                    ).format(raw),
                    (travel_mode, hex_id_7, observed_at, observed_at),
                )
                history = [
                    {"at": _utc(at), "demand": int(value)} for at, value in cursor.fetchall()
                ]

            cursor.execute(
                sql.SQL(
                    "SELECT horizon_minutes, predicted_demand FROM {} "
                    "WHERE travel_mode = %s AND hex_id_7 = %s "
                    "AND forecast_start_utc = %s AND model_name = 'lightgbm' "
                    "ORDER BY horizon_minutes"
                ).format(predictions),
                (travel_mode, hex_id_7, forecast_start_utc),
            )
            horizons = {str(minutes): round(float(value), 2) for minutes, value in cursor.fetchall()}

    return {"hex_id_7": hex_id_7, "history": history, "horizons": horizons}
