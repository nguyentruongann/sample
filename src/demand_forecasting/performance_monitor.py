"""Delayed hourly performance windows, retryable and separated from scraping."""
import argparse
import json
import pandas as pd
from .monitoring import prepare,score,serializable,KEYS
from .monitoring_config import MonitorConfig


def due_windows(now,config,timezone='Asia/Ho_Chi_Minh'):
    local=pd.Timestamp(now).tz_convert(timezone)
    anchor=local.normalize()+pd.Timedelta(hours=(local.hour//config.interval_hours)*config.interval_hours)
    end=anchor-pd.Timedelta(hours=config.delay_hours)
    count=max(1,config.recheck_hours//config.interval_hours)
    return [(end-pd.Timedelta(hours=(i+1)*config.interval_hours),end-pd.Timedelta(hours=i*config.interval_hours))
            for i in reversed(range(count))]


def evaluate_window(frame,start,end,now,config):
    if not frame.empty:
        frame=frame.loc[(frame.forecast_start_utc>=start)&(frame.forecast_start_utc<end)]
    prepared=prepare(frame,now,config.label_wait_seconds,config.issue_grace_seconds)
    rows=[]
    for mode in ('Car','Motorcycle'):
        for horizon in (10,30,60):
            group=prepared.loc[(prepared.travel_mode==mode)&(prepared.horizon_minutes==horizon)]
            groups=list(group.groupby('model_version')) if len(group) else [('unknown',group)]
            for version,part in groups:
                eligible=part.loc[part.eligible];metrics=score(eligible)
                complete=int((part.mature&part.complete).sum());mature=int(part.mature.sum())
                missing=int((part.mature&~part.complete).sum());pending=int((~part.mature).sum())
                late=int((~part.on_time).sum())
                status=('no_predictions' if not len(part) else 'pending_actual' if pending else
                        'partial_actual' if missing else 'no_eligible_predictions' if not len(eligible) else
                        'insufficient_samples' if metrics['samples']<config.min_samples or metrics['actual_sum']<config.min_demand else 'ready')
                rows.append(dict(window_start=start.isoformat(),window_end=end.isoformat(),
                    travel_mode=mode,horizon_minutes=horizon,model_version=str(version),**metrics,
                    prediction_sum=float(eligible.predicted_demand.sum()),
                    forecasts=len(part),pending=pending,missing_actual=missing,late=late,
                    actual_coverage_pct=100*complete/mature if mature else None,
                    forecast_origins=int(part.forecast_start_utc.nunique()),
                    window_forecast_origins=int(group.forecast_start_utc.nunique()),
                    expected_origins=int((end-start).total_seconds()/600),
                    status=status,alert_ready=status=='ready'))
    return serializable(rows)


def run(now=None,settings=None,config=None):
    import psycopg
    from .config import get_settings
    from .monitoring_store import ensure_tables,load_evaluation,save_windows,write_state,table
    from psycopg import sql
    settings=settings or get_settings();config=config or MonitorConfig.from_env()
    now=pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    if now.tzinfo is None: raise ValueError('Evaluation time must include timezone')
    with psycopg.connect(settings.require_database_url(),connect_timeout=5) as conn:
        ensure_tables(conn,settings)
        # Serialize concurrent CLI/DAG evaluations; short migration lock is released first.
    with psycopg.connect(settings.require_database_url(),connect_timeout=5) as conn:
        if not conn.execute('SELECT pg_try_advisory_xact_lock(20261006)').fetchone()[0]:
            return {'status':'already_running'}
        first=conn.execute(sql.SQL('SELECT min(forecast_start_utc) FROM {}').format(table(settings,'monitor_prediction_archive'))).fetchone()[0]
        windows=due_windows(now,config,settings.local_timezone)
        if first is not None: windows=[w for w in windows if w[1]>pd.Timestamp(first)]
        else: windows=windows[-1:]
        done=[]
        for start,end in windows:
            frame=load_evaluation(settings,start,end)
            rows=evaluate_window(frame,start,end,now,config)
            save_windows(conn,settings,start,end,rows);done.append(dict(start=start.isoformat(),end=end.isoformat()))
        result=dict(status='success',evaluated_at=now.isoformat(),windows=done,
                    interval_hours=config.interval_hours,delay_hours=config.delay_hours)
        write_state(conn,settings,'performance_job',result)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--as-of',help='Timezone-aware evaluation time; for replay/testing')
    args=parser.parse_args();print(json.dumps(run(now=args.as_of),indent=2))
