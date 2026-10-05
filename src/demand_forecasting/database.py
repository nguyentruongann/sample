"""PostgreSQL storage using transactional DDL and streaming COPY."""

from __future__ import annotations

import io
import re
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import Settings, get_settings

RAW_COLUMNS = [
    "period_datetime_utc",
    "hex_id_7",
    "travel_mode",
    "total_demand",
    "is_holiday",
]

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class LoadSummary:
    parquet_path: Path
    schema: str
    table: str
    expected_rows: int
    loaded_rows: int
    elapsed_seconds: float


def _validate_identifier(value: str, label: str) -> str:
    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"Invalid PostgreSQL {label}: {value!r}")
    return value


def sqlalchemy_url(database_url: str) -> str:
    if database_url.startswith("postgres://"):
        database_url = "postgresql://" + database_url[len("postgres://") :]
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return database_url


def wait_for_database(
    settings: Settings | None = None,
    retries: int = 30,
    delay_seconds: float = 2.0,
) -> None:
    """Wait until PostgreSQL accepts a connection."""

    import psycopg

    settings = settings or get_settings()
    database_url = settings.require_database_url()
    for attempt in range(retries):
        try:
            with psycopg.connect(database_url, connect_timeout=5):
                return
        except psycopg.OperationalError as error:
            detail = str(error).lower()
            if "does not exist" in detail:
                raise RuntimeError(
                    "PostgreSQL is reachable, but the requested database/role does not exist. "
                    "Run init-runtime with POSTGRES_ADMIN_URL configured first."
                ) from None
            if "password authentication failed" in detail:
                raise RuntimeError("PostgreSQL rejected DATABASE_URL credentials; check .env") from None
            if attempt + 1 < retries:
                time.sleep(delay_seconds)

    raise RuntimeError(
        f"PostgreSQL is unreachable after {retries} attempts; check the host/port, "
        "listen_addresses, pg_hba.conf and firewall."
    ) from None


def copy_parquet_to_postgres(
    parquet_path: str | Path,
    settings: Settings | None = None,
    batch_size: int = 200_000,
) -> LoadSummary:
    """Replace the raw table atomically and stream a Parquet file with COPY.

    PostgreSQL DDL is transactional. If COPY or validation fails, the previous
    table remains available after rollback.
    """

    import psycopg
    import pyarrow.parquet as pq
    from psycopg import sql

    settings = settings or get_settings()
    database_url = settings.require_database_url()
    schema = _validate_identifier(settings.postgres_schema, "schema")
    table = _validate_identifier(settings.postgres_table, "table")
    parquet_path = Path(parquet_path).expanduser().resolve()

    if not parquet_path.is_file():
        raise FileNotFoundError(parquet_path)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    parquet = pq.ParquetFile(parquet_path)
    missing = sorted(set(RAW_COLUMNS) - set(parquet.schema_arrow.names))
    if missing:
        raise ValueError(f"Parquet is missing required columns: {missing}")

    expected_rows = parquet.metadata.num_rows
    loaded_rows = 0
    started = time.perf_counter()

    qualified_table = sql.SQL("{}.{}").format(sql.Identifier(schema), sql.Identifier(table))
    pk_name = _validate_identifier(f"{table}_pk", "constraint")
    time_index = _validate_identifier(f"idx_{table}_time", "index")
    mode_time_index = _validate_identifier(f"idx_{table}_mode_time", "index")

    with psycopg.connect(database_url) as connection, connection.cursor() as cursor:
        cursor.execute("SET TIME ZONE 'UTC'")
        cursor.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(qualified_table))
        cursor.execute(
            sql.SQL(
                """
                    CREATE TABLE {} (
                        period_datetime_utc TIMESTAMPTZ NOT NULL,
                        hex_id_7 TEXT NOT NULL,
                        travel_mode VARCHAR(16) NOT NULL
                            CHECK (travel_mode IN ('Car', 'Motorcycle')),
                        total_demand SMALLINT NOT NULL CHECK (total_demand >= 0),
                        is_holiday SMALLINT NOT NULL CHECK (is_holiday IN (0, 1))
                    )
                    """
            ).format(qualified_table)
        )

        copy_query = sql.SQL("COPY {} ({}) FROM STDIN WITH (FORMAT CSV, NULL '\\N')").format(
            qualified_table,
            sql.SQL(", ").join(map(sql.Identifier, RAW_COLUMNS)),
        )

        with cursor.copy(copy_query) as copy:
            for record_batch in parquet.iter_batches(
                batch_size=batch_size,
                columns=RAW_COLUMNS,
            ):
                chunk = record_batch.to_pandas()[RAW_COLUMNS]
                buffer = io.StringIO()
                chunk.to_csv(
                    buffer,
                    index=False,
                    header=False,
                    na_rep=r"\N",
                    date_format="%Y-%m-%d %H:%M:%S.%f%z",
                    lineterminator="\n",
                )
                copy.write(buffer.getvalue().encode("utf-8"))
                loaded_rows += len(chunk)

        if loaded_rows != expected_rows:
            raise RuntimeError(
                f"COPY row mismatch: expected={expected_rows}, streamed={loaded_rows}"
            )

        cursor.execute(
            sql.SQL(
                "ALTER TABLE {} ADD CONSTRAINT {} "
                "PRIMARY KEY (travel_mode, hex_id_7, period_datetime_utc)"
            ).format(qualified_table, sql.Identifier(pk_name))
        )
        cursor.execute(
            sql.SQL("CREATE INDEX {} ON {} (period_datetime_utc)").format(
                sql.Identifier(time_index), qualified_table
            )
        )
        cursor.execute(
            sql.SQL("CREATE INDEX {} ON {} (travel_mode, period_datetime_utc)").format(
                sql.Identifier(mode_time_index), qualified_table
            )
        )
        cursor.execute(sql.SQL("ANALYZE {}").format(qualified_table))
        cursor.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(qualified_table))
        database_rows = int(cursor.fetchone()[0])

        if database_rows != expected_rows:
            raise RuntimeError(
                f"Database row mismatch: expected={expected_rows}, actual={database_rows}"
            )

    return LoadSummary(
        parquet_path=parquet_path,
        schema=schema,
        table=table,
        expected_rows=expected_rows,
        loaded_rows=database_rows,
        elapsed_seconds=time.perf_counter() - started,
    )


def load_raw_demand(
    settings: Settings | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load raw demand from PostgreSQL into pandas."""

    from sqlalchemy import create_engine

    settings = settings or get_settings()
    schema = _validate_identifier(settings.postgres_schema, "schema")
    table = _validate_identifier(settings.postgres_table, "table")

    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    query = f"SELECT {', '.join(RAW_COLUMNS)} FROM {schema}.{table}"
    if limit is not None:
        query += f" LIMIT {int(limit)}"

    engine = create_engine(sqlalchemy_url(settings.require_database_url()))
    try:
        frame = pd.read_sql_query(query, con=engine)
    finally:
        engine.dispose()

    frame["period_datetime_utc"] = pd.to_datetime(
        frame["period_datetime_utc"], utc=True, errors="raise"
    )
    frame["total_demand"] = frame["total_demand"].astype("int16")
    frame["is_holiday"] = frame["is_holiday"].astype("int8")
    return frame


def database_row_count(settings: Settings | None = None) -> int:
    import psycopg
    from psycopg import sql

    settings = settings or get_settings()
    schema = _validate_identifier(settings.postgres_schema, "schema")
    table = _validate_identifier(settings.postgres_table, "table")

    with psycopg.connect(settings.require_database_url()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("SELECT COUNT(*) FROM {}.{}").format(
                    sql.Identifier(schema), sql.Identifier(table)
                )
            )
            return int(cursor.fetchone()[0])


def database_has_rows(settings: Settings | None = None) -> bool:
    """Fast non-destructive check before generating historical fake demand."""
    import psycopg
    from psycopg import sql

    settings = settings or get_settings()
    schema = _validate_identifier(settings.postgres_schema, "schema")
    table = _validate_identifier(settings.postgres_table, "table")
    with psycopg.connect(settings.require_database_url()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("SELECT EXISTS (SELECT 1 FROM {}.{} LIMIT 1)").format(
                    sql.Identifier(schema), sql.Identifier(table)
                )
            )
            return bool(cursor.fetchone()[0])


def ensure_runtime_tables(settings: Settings | None = None) -> None:
    """Create the append-only live input and prediction tables when absent."""

    import psycopg
    from psycopg import sql

    settings = settings or get_settings()
    schema = _validate_identifier(settings.postgres_schema, "schema")
    raw_table = _validate_identifier(settings.postgres_table, "table")
    predictions_table = _validate_identifier(settings.predictions_table, "table")

    raw = sql.SQL("{}.{}").format(sql.Identifier(schema), sql.Identifier(raw_table))
    predictions = sql.SQL("{}.{}").format(sql.Identifier(schema), sql.Identifier(predictions_table))

    with psycopg.connect(settings.require_database_url()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
            cursor.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {} (
                        period_datetime_utc TIMESTAMPTZ NOT NULL,
                        hex_id_7 TEXT NOT NULL,
                        travel_mode VARCHAR(16) NOT NULL
                            CHECK (travel_mode IN ('Car', 'Motorcycle')),
                        total_demand SMALLINT NOT NULL CHECK (total_demand >= 0),
                        is_holiday SMALLINT NOT NULL CHECK (is_holiday IN (0, 1)),
                        PRIMARY KEY (travel_mode, hex_id_7, period_datetime_utc)
                    )
                    """
                ).format(raw)
            )
            cursor.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (period_datetime_utc)").format(
                    sql.Identifier(f"idx_{raw_table}_time"), raw
                )
            )
            cursor.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {} (
                        forecast_start_utc TIMESTAMPTZ NOT NULL,
                        hex_id_7 TEXT NOT NULL,
                        travel_mode VARCHAR(16) NOT NULL
                            CHECK (travel_mode IN ('Car', 'Motorcycle')),
                        horizon_minutes SMALLINT NOT NULL CHECK (horizon_minutes > 0),
                        model_name TEXT NOT NULL,
                        model_version TEXT NOT NULL,
                        predicted_demand DOUBLE PRECISION NOT NULL
                            CHECK (predicted_demand >= 0),
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (
                            forecast_start_utc,
                            hex_id_7,
                            travel_mode,
                            horizon_minutes,
                            model_name
                        )
                    )
                    """
                ).format(predictions)
            )
            cursor.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {} ON {} "
                    "(forecast_start_utc DESC, travel_mode, horizon_minutes)"
                ).format(sql.Identifier(f"idx_{predictions_table}_latest"), predictions)
            )


def upsert_raw_demand(frame: pd.DataFrame, settings: Settings | None = None) -> int:
    """Insert missing fake rows without replacing observations already in PostgreSQL."""

    import psycopg
    from psycopg import sql

    settings = settings or get_settings()
    ensure_runtime_tables(settings)
    schema = _validate_identifier(settings.postgres_schema, "schema")
    table = _validate_identifier(settings.postgres_table, "table")

    missing = sorted(set(RAW_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Raw frame is missing required columns: {missing}")
    if frame.empty:
        return 0

    rows = frame[RAW_COLUMNS].copy()
    rows["period_datetime_utc"] = pd.to_datetime(
        rows["period_datetime_utc"], utc=True, errors="raise"
    )
    rows["hex_id_7"] = rows["hex_id_7"].astype(str)
    rows["total_demand"] = pd.to_numeric(rows["total_demand"], errors="raise").astype(int)
    rows["is_holiday"] = pd.to_numeric(rows["is_holiday"], errors="raise").astype(int)

    records = list(rows.itertuples(index=False, name=None))
    qualified = sql.SQL("{}.{}").format(sql.Identifier(schema), sql.Identifier(table))
    query = sql.SQL(
        """
        INSERT INTO {} ({}) VALUES ({})
        ON CONFLICT (travel_mode, hex_id_7, period_datetime_utc) DO NOTHING
        """
    ).format(
        qualified,
        sql.SQL(", ").join(map(sql.Identifier, RAW_COLUMNS)),
        sql.SQL(", ").join(sql.Placeholder() for _ in RAW_COLUMNS),
    )

    with psycopg.connect(settings.require_database_url()) as connection:
        with connection.cursor() as cursor:
            cursor.executemany(query, records)
            return cursor.rowcount


def live_bucket_row_counts(start_utc, end_utc, settings: Settings | None = None) -> dict[pd.Timestamp, int]:
    """Count rows in the generated live era; historical notebook data is sparse by design."""
    import psycopg
    from psycopg import sql

    settings = settings or get_settings()
    schema = _validate_identifier(settings.postgres_schema, "schema")
    table = _validate_identifier(settings.postgres_table, "table")
    query = sql.SQL(
        "SELECT period_datetime_utc, COUNT(*) FROM {}.{} "
        "WHERE period_datetime_utc >= %s AND period_datetime_utc < %s "
        "GROUP BY period_datetime_utc"
    ).format(sql.Identifier(schema), sql.Identifier(table))
    with psycopg.connect(settings.require_database_url()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, (pd.Timestamp(start_utc).to_pydatetime(),
                                   pd.Timestamp(end_utc).to_pydatetime()))
            return {pd.Timestamp(stamp).tz_convert("UTC"): int(count)
                    for stamp, count in cursor.fetchall()}


def ensure_live_generator_version(
    version: str, live_start_utc, settings: Settings | None = None,
) -> bool:
    """Once per generator version, rebuild only the synthetic live-era rows."""
    import psycopg
    from psycopg import sql

    settings = settings or get_settings()
    schema = _validate_identifier(settings.postgres_schema, "schema")
    raw_table = _validate_identifier(settings.postgres_table, "table")
    metadata = sql.SQL("{}.{}").format(sql.Identifier(schema), sql.Identifier("demand_runtime_metadata"))
    raw = sql.SQL("{}.{}").format(sql.Identifier(schema), sql.Identifier(raw_table))
    cutoff = pd.Timestamp(live_start_utc).tz_convert("UTC").to_pydatetime()
    with psycopg.connect(settings.require_database_url()) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(%s)", (20260926,))
            cursor.execute(sql.SQL(
                "CREATE TABLE IF NOT EXISTS {} (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            ).format(metadata))
            cursor.execute(sql.SQL("SELECT value FROM {} WHERE key = %s").format(metadata),
                           ("live_generator_version",))
            row = cursor.fetchone()
            if row is not None and row[0] == version:
                return False
            # This project owns the fake_demand_10min table. Keep the notebook
            # historical period; replace synthetic extensions built by older code.
            cursor.execute(sql.SQL("DELETE FROM {} WHERE period_datetime_utc >= %s").format(raw),
                           (cutoff,))
            cursor.execute(sql.SQL(
                "INSERT INTO {} (key, value) VALUES (%s, %s) "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value"
            ).format(metadata), ("live_generator_version", version))
            return True


def load_demand_history(
    start_utc,
    end_utc=None,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Read an exact UTC window for online feature construction."""

    from sqlalchemy import create_engine, text

    settings = settings or get_settings()
    schema = _validate_identifier(settings.postgres_schema, "schema")
    table = _validate_identifier(settings.postgres_table, "table")
    start = pd.Timestamp(start_utc)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")

    clauses = ["period_datetime_utc >= :start_utc"]
    params = {"start_utc": start.to_pydatetime()}
    if end_utc is not None:
        end = pd.Timestamp(end_utc)
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        clauses.append("period_datetime_utc < :end_utc")
        params["end_utc"] = end.to_pydatetime()

    query = text(
        f"SELECT {', '.join(RAW_COLUMNS)} FROM {schema}.{table} "
        f"WHERE {' AND '.join(clauses)} ORDER BY period_datetime_utc, travel_mode, hex_id_7"
    )
    engine = create_engine(sqlalchemy_url(settings.require_database_url()))
    try:
        frame = pd.read_sql_query(query, con=engine, params=params)
    finally:
        engine.dispose()

    if not frame.empty:
        frame["period_datetime_utc"] = pd.to_datetime(
            frame["period_datetime_utc"], utc=True, errors="raise"
        )
        frame["total_demand"] = frame["total_demand"].astype("int16")
        frame["is_holiday"] = frame["is_holiday"].astype("int8")
    return frame


def latest_observation_timestamp(settings: Settings | None = None) -> pd.Timestamp | None:
    import psycopg
    from psycopg import sql

    settings = settings or get_settings()
    schema = _validate_identifier(settings.postgres_schema, "schema")
    table = _validate_identifier(settings.postgres_table, "table")
    with psycopg.connect(settings.require_database_url()) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("SELECT MAX(period_datetime_utc) FROM {}.{}").format(
                    sql.Identifier(schema), sql.Identifier(table)
                )
            )
            value = cursor.fetchone()[0]
    return pd.Timestamp(value).tz_convert("UTC") if value is not None else None


PREDICTION_COLUMNS = [
    "forecast_start_utc",
    "hex_id_7",
    "travel_mode",
    "horizon_minutes",
    "model_name",
    "model_version",
    "predicted_demand",
]


def upsert_predictions(frame: pd.DataFrame, settings: Settings | None = None) -> int:
    """Write inference results idempotently for a scheduled forecast start."""

    import psycopg
    from psycopg import sql

    settings = settings or get_settings()
    ensure_runtime_tables(settings)
    schema = _validate_identifier(settings.postgres_schema, "schema")
    table = _validate_identifier(settings.predictions_table, "table")
    missing = sorted(set(PREDICTION_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction frame is missing required columns: {missing}")
    if frame.empty:
        return 0

    rows = frame[PREDICTION_COLUMNS].copy()
    rows["forecast_start_utc"] = pd.to_datetime(
        rows["forecast_start_utc"], utc=True, errors="raise"
    )
    rows["predicted_demand"] = pd.to_numeric(rows["predicted_demand"], errors="raise").clip(lower=0)
    records = list(rows.itertuples(index=False, name=None))

    qualified = sql.SQL("{}.{}").format(sql.Identifier(schema), sql.Identifier(table))
    query = sql.SQL(
        """
        INSERT INTO {} ({}) VALUES ({})
        ON CONFLICT (
            forecast_start_utc, hex_id_7, travel_mode, horizon_minutes, model_name
        ) DO UPDATE SET
            model_version = EXCLUDED.model_version,
            predicted_demand = EXCLUDED.predicted_demand,
            created_at = NOW()
        """
    ).format(
        qualified,
        sql.SQL(", ").join(map(sql.Identifier, PREDICTION_COLUMNS)),
        sql.SQL(", ").join(sql.Placeholder() for _ in PREDICTION_COLUMNS),
    )
    with psycopg.connect(settings.require_database_url()) as connection:
        with connection.cursor() as cursor:
            cursor.executemany(query, records)
        from .monitoring_store import archive_predictions
        archive_predictions(connection, settings, records, frame.attrs.get("monitor_health"))
    return len(records)


def load_latest_predictions(
    settings: Settings | None = None,
    travel_mode: str | None = None,
    horizon_minutes: int | None = None,
    hex_id_7: str | None = None,
    limit: int = 1000,
) -> pd.DataFrame:
    """Return the newest forecast batch with optional API filters."""

    from sqlalchemy import create_engine, text

    settings = settings or get_settings()
    schema = _validate_identifier(settings.postgres_schema, "schema")
    table = _validate_identifier(settings.predictions_table, "table")
    limit = max(1, min(int(limit), 10_000))
    clauses = [f"forecast_start_utc = (SELECT MAX(forecast_start_utc) FROM {schema}.{table})"]
    params: dict[str, object] = {"limit": limit}
    if travel_mode is not None:
        clauses.append("travel_mode = :travel_mode")
        params["travel_mode"] = travel_mode
    if horizon_minutes is not None:
        clauses.append("horizon_minutes = :horizon_minutes")
        params["horizon_minutes"] = int(horizon_minutes)
    if hex_id_7 is not None:
        clauses.append("hex_id_7 = :hex_id_7")
        params["hex_id_7"] = hex_id_7

    query = text(
        f"SELECT {', '.join(PREDICTION_COLUMNS)}, created_at "
        f"FROM {schema}.{table} WHERE {' AND '.join(clauses)} "
        "ORDER BY travel_mode, horizon_minutes, predicted_demand DESC LIMIT :limit"
    )
    engine = create_engine(sqlalchemy_url(settings.require_database_url()))
    try:
        return pd.read_sql_query(query, con=engine, params=params)
    finally:
        engine.dispose()


def database_is_ready(settings: Settings | None = None) -> bool:
    import psycopg

    settings = settings or get_settings()
    try:
        with psycopg.connect(settings.require_database_url(), connect_timeout=3) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                return cursor.fetchone()[0] == 1
    except psycopg.Error:
        return False
