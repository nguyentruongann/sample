"""Transactional monitoring archive. Existing serving table remains compatible."""
from __future__ import annotations
import json
from .database import _validate_identifier


def table(settings, name):
    from psycopg import sql
    return sql.SQL('{}.{}').format(sql.Identifier(_validate_identifier(settings.postgres_schema, 'schema')),
                                  sql.Identifier(name))


def ensure_tables(connection, settings):
    from psycopg import sql
    with connection.cursor() as c:
        c.execute('SELECT pg_advisory_xact_lock(20261005)')
        c.execute(sql.SQL('CREATE SCHEMA IF NOT EXISTS {}').format(sql.Identifier(settings.postgres_schema)))
        c.execute(sql.SQL('''CREATE TABLE IF NOT EXISTS {} (
            forecast_start_utc timestamptz NOT NULL, hex_id_7 text NOT NULL,
            travel_mode text NOT NULL, horizon_minutes smallint NOT NULL,
            model_name text NOT NULL, model_version text NOT NULL,
            predicted_demand double precision NOT NULL CHECK(predicted_demand >= 0),
            issued_at timestamptz NOT NULL DEFAULT clock_timestamp(),
            PRIMARY KEY(forecast_start_utc,hex_id_7,travel_mode,horizon_minutes,model_name)
        )''').format(table(settings, 'monitor_prediction_archive')))
        c.execute(sql.SQL('''CREATE TABLE IF NOT EXISTS {} (
            key text PRIMARY KEY, payload jsonb NOT NULL, updated_at timestamptz NOT NULL DEFAULT now()
        )''').format(table(settings, 'monitor_state')))

        c.execute(sql.SQL("""CREATE TABLE IF NOT EXISTS {} (
            window_start timestamptz NOT NULL, window_end timestamptz NOT NULL,
            travel_mode text NOT NULL, horizon_minutes smallint NOT NULL,
            model_version text NOT NULL, payload jsonb NOT NULL,
            evaluated_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY(window_start,window_end,travel_mode,horizon_minutes,model_version)
        )""").format(table(settings,'monitor_performance_windows')))


def write_state(connection, settings, key, payload):
    from psycopg import sql
    from psycopg.types.json import Jsonb
    with connection.cursor() as c:
        c.execute(sql.SQL('''INSERT INTO {} (key,payload) VALUES (%s,%s)
            ON CONFLICT(key) DO UPDATE SET payload=EXCLUDED.payload,updated_at=now()''')
            .format(table(settings, 'monitor_state')), (key, Jsonb(payload)))


def archive_predictions(connection, settings, records, health):
    from psycopg import sql
    ensure_tables(connection, settings)
    with connection.cursor() as c:
        c.executemany(sql.SQL('''INSERT INTO {} (forecast_start_utc,hex_id_7,travel_mode,
            horizon_minutes,model_name,model_version,predicted_demand)
            VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING''')
            .format(table(settings, 'monitor_prediction_archive')), records)
    if health:
        write_state(connection, settings, 'inference', health)


def read_state(settings, key='report'):
    import psycopg
    from psycopg import sql
    with psycopg.connect(settings.require_database_url(), connect_timeout=5) as conn:
        with conn.cursor() as c:
            c.execute(sql.SQL('SELECT payload,updated_at FROM {} WHERE key=%s')
                      .format(table(settings,'monitor_state')), (key,))
            row=c.fetchone()
    return None if row is None else dict(row[0], updated_at=row[1].isoformat())


def load_evaluation(settings, start, end):
    """Exact [T,T+H) join; missing buckets remain NULL, never zero.

    Recomputed by the scheduled validation job so late arrivals are incorporated. Archive
    starts only after installation: do not import historical overwritten forecasts.
    """
    import os
    import pandas as pd
    import psycopg
    from psycopg import sql
    params = [start, end]
    limit=int(os.getenv('MONITOR_MAX_ROWS','1000000'))
    params.append(limit+1)
    query=sql.SQL('''SELECT p.*, a.bucket_count, a.actual
        FROM {} p LEFT JOIN LATERAL (
          SELECT count(*) AS bucket_count, sum(r.total_demand)::float8 AS actual
          FROM {} r WHERE r.travel_mode=p.travel_mode AND r.hex_id_7=p.hex_id_7
          AND r.period_datetime_utc >= p.forecast_start_utc
          AND r.period_datetime_utc < p.forecast_start_utc + p.horizon_minutes * interval '1 minute'
          AND mod(extract(epoch from (r.period_datetime_utc-p.forecast_start_utc))::numeric,600)=0
        ) a ON true WHERE p.forecast_start_utc >= %s AND p.forecast_start_utc < %s
        ORDER BY p.forecast_start_utc LIMIT %s''').format(
            table(settings,'monitor_prediction_archive'),table(settings,settings.postgres_table))
    with psycopg.connect(settings.require_database_url(),connect_timeout=5) as conn:
        conn.execute("SET statement_timeout='120s'")
        with conn.cursor() as c:
            c.execute(query,params)
            frame=pd.DataFrame(c.fetchall(),columns=[x.name for x in c.description])
    if len(frame)>limit:
        raise RuntimeError('Monitoring row budget exceeded; increase MONITOR_MAX_ROWS or partition aggregation')
    return frame


def monitored_job(name):
    """Decorator for Airflow task bodies; telemetry failure never masks job errors."""
    import functools
    import logging
    import time
    from .config import get_settings

    def decorate(fn):
        @functools.wraps(fn)
        def run(*args,**kwargs):
            started=time.time()
            def save(status):
                try:
                    import psycopg
                    settings=get_settings()
                    with psycopg.connect(settings.require_database_url(),connect_timeout=5) as conn:
                        conn.execute("SET statement_timeout='5s'")
                        ensure_tables(conn,settings)
                        write_state(conn,settings,'job:'+name,dict(name=name,status=status,
                            started_at=started,finished_at=time.time() if status!='running' else None,
                            duration_seconds=time.time()-started))
                except Exception:
                    logging.getLogger(__name__).warning('Could not record job monitoring status: %s',name)
            save('running')
            try:
                result=fn(*args,**kwargs)
            except BaseException:
                save('failed'); raise
            save('success'); return result
        return run
    return decorate


def save_windows(connection,settings,start,end,rows):
    from psycopg import sql
    from psycopg.types.json import Jsonb
    # Replace one window atomically, including removal of old 'no_predictions' placeholders.
    with connection.cursor() as c:
        c.execute(sql.SQL('DELETE FROM {} WHERE window_start=%s AND window_end=%s')
                  .format(table(settings,'monitor_performance_windows')),(start,end))
        c.executemany(sql.SQL('''INSERT INTO {} (window_start,window_end,travel_mode,horizon_minutes,
                  model_version,payload) VALUES (%s,%s,%s,%s,%s,%s)''')
                  .format(table(settings,'monitor_performance_windows')),
                  [(start,end,r['travel_mode'],r['horizon_minutes'],r['model_version'],Jsonb(r)) for r in rows])


def load_windows(settings,hours=72,as_of=None):
    from datetime import datetime,timezone
    anchor=as_of if as_of is not None else datetime.now(timezone.utc)
    import psycopg
    from psycopg import sql
    with psycopg.connect(settings.require_database_url(),connect_timeout=5) as conn:
        with conn.cursor() as c:
            c.execute(sql.SQL('SELECT payload,evaluated_at FROM {} WHERE window_end>=%s-%s*interval \'1 hour\' AND window_end<=%s ORDER BY window_end DESC,travel_mode,horizon_minutes')
                .format(table(settings,'monitor_performance_windows')),(anchor,hours,anchor))
            return [dict(payload,evaluated_at=stamp.isoformat()) for payload,stamp in c.fetchall()]


def load_serving(settings,start,end):
    import pandas as pd
    import psycopg
    from psycopg import sql
    from .monitoring_config import MonitorConfig
    limit=MonitorConfig.from_env().max_rows
    with psycopg.connect(settings.require_database_url(),connect_timeout=5) as conn:
        conn.execute("SET statement_timeout='60s'")
        with conn.cursor() as c:
            c.execute(sql.SQL('''SELECT forecast_start_utc,issued_at,travel_mode,horizon_minutes,
                model_version,predicted_demand FROM {} WHERE issued_at >= %s AND issued_at < %s
                ORDER BY issued_at LIMIT %s''').format(table(settings,'monitor_prediction_archive')),(start,end,limit+1))
            frame=pd.DataFrame(c.fetchall(),columns=[d.name for d in c.description])
    if len(frame)>limit: raise RuntimeError('Serving row budget exceeded')
    return frame


def load_states(settings):
    import psycopg
    from psycopg import sql
    with psycopg.connect(settings.require_database_url(),connect_timeout=5) as conn:
        with conn.cursor() as c:
            c.execute(sql.SQL('SELECT key,payload,updated_at FROM {}').format(table(settings,'monitor_state')))
            return {key:dict(payload,updated_at=stamp.isoformat()) for key,payload,stamp in c.fetchall()}
