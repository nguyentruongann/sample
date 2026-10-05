"""OPTIONAL accelerated replay. Production data, model files and clocks are untouched."""
from __future__ import annotations
import argparse
import json
import uuid
from dataclasses import replace
import pandas as pd
from psycopg import sql


def demo_schema_prefix(settings):
    return settings.postgres_schema[:32]+'_monitor_demo_'


def validate_demo_schema(settings,schema):
    import re
    if not re.fullmatch(re.escape(demo_schema_prefix(settings))+r'[0-9a-f]{8}',schema):
        raise ValueError('Only isolated monitoring demo schemas are allowed')
    return schema


def _replay_snapshot(settings,hours=3,start=None,progress=print):
    import psycopg
    import joblib
    from .database import ensure_runtime_tables,upsert_raw_demand
    from .incremental_data import generate_live_bucket
    from .predict import predict_and_store
    from .monitoring_config import MonitorConfig
    from .monitoring_store import table,ensure_tables,write_state
    from .performance_monitor import run
    from .monitoring_service import collect
    if hours<1 or hours>48: raise ValueError('--hours must be 1..48')
    c=MonitorConfig.from_env()
    if hours % c.interval_hours: raise ValueError('--hours must be a multiple of MONITOR_INTERVAL_HOURS')
    now=pd.Timestamp.now(tz='UTC')
    begin=pd.Timestamp(start) if start else now.tz_convert(settings.local_timezone).floor(f'{c.interval_hours}h')
    if begin.tzinfo is None: raise ValueError('--start must include timezone')
    local=begin.tz_convert(settings.local_timezone)
    if local.minute or local.second or local.microsecond or local.hour%c.interval_hours:
        raise ValueError('--start must align to configured evaluation hour boundary')
    manifest=json.loads((settings.model_dir/'active.json').read_text())
    for mode in ('Car','Motorcycle'):
        for h in (10,30,60):
            relative=manifest.get('model_files',{}).get(mode,{}).get(str(h))
            if not relative: raise ValueError('Replay requires all six active models')
            path=(settings.model_dir/relative).resolve()
            if not path.is_relative_to(settings.model_dir.resolve()):raise ValueError('Invalid model path')
            artifact=joblib.load(path)
            if begin<pd.Timestamp(artifact['test_start']):
                raise ValueError('Replay start must be after final model fitting period')
    schema=demo_schema_prefix(settings)+uuid.uuid4().hex[:8]
    validate_demo_schema(settings,schema)
    demo=replace(settings,postgres_schema=schema)
    ensure_runtime_tables(demo)
    with psycopg.connect(demo.require_database_url()) as conn:ensure_tables(conn,demo)
    end=begin+pd.Timedelta(hours=hours)
    warm_start=begin-pd.Timedelta(days=settings.prediction_history_days)
    progress(f'DEMO schema={schema}; seed {settings.prediction_history_days} days of lag history (no sleep)')
    # Seed only pre-origin history, then append one CLOSED bucket per virtual step.
    pending=[]
    for stamp in pd.date_range(warm_start,begin,freq='10min',inclusive='left'):
        pending.append(generate_live_bucket(stamp))
        if len(pending)>=36:
            upsert_raw_demand(pd.concat(pending,ignore_index=True),demo);pending=[]
    if pending:upsert_raw_demand(pd.concat(pending,ignore_index=True),demo)
    for i,t in enumerate(pd.date_range(begin,end,freq='10min',inclusive='left')):
        if i:upsert_raw_demand(generate_live_bucket(t-pd.Timedelta(minutes=10)),demo)
        predict_and_store(forecast_start_utc=t,settings=demo)
        # Virtual timestamps are ONLY written to this freshly created, validated schema.
        # This intentionally never adds a simulated-time option to production inference.
        with psycopg.connect(demo.require_database_url()) as conn:
            conn.execute(sql.SQL('UPDATE {} SET issued_at=%s WHERE forecast_start_utc=%s')
                         .format(table(demo,'monitor_prediction_archive')),(t+pd.Timedelta(seconds=1),t))
            conn.execute(sql.SQL("UPDATE {} SET payload=jsonb_set(payload,'{{observed_at}}',to_jsonb(%s::text)) WHERE key LIKE 'feature:%%'")
                         .format(table(demo,'monitor_state')),(t.isoformat(),))
        progress(f'DEMO {i+1}/{hours*6}: {t.isoformat()}')
    virtual_now=end+pd.Timedelta(hours=c.delay_hours)
    # Drain labels through the final H60 target, then add configured label-wait if needed.
    required_now=end+pd.Timedelta(minutes=50,seconds=c.label_wait_seconds)
    if virtual_now<required_now:
        raise ValueError('Increase delay hours or reduce label wait for replay maturity')
    last_loaded=end-pd.Timedelta(minutes=20) if hours*6>1 else begin-pd.Timedelta(minutes=10)
    pending=[]
    for t in pd.date_range(last_loaded+pd.Timedelta(minutes=10),virtual_now,freq='10min',inclusive='left'):
        pending.append(generate_live_bucket(t))
    if pending:upsert_raw_demand(pd.concat(pending,ignore_index=True),demo)
    for anchor in pd.date_range(begin+pd.Timedelta(hours=c.interval_hours+c.delay_hours),virtual_now,
                                freq=f'{c.interval_hours}h'):
        run(anchor,settings=demo,config=c)
    report=collect(settings=demo,now=virtual_now,config=c)
    report['demo']=dict(schema=schema,virtual_now=virtual_now.isoformat(),hours=hours,
        note='Simulation only; feature snapshot may be stale after waiting virtually for H60 labels.')
    with psycopg.connect(demo.require_database_url()) as conn:write_state(conn,demo,'demo_report',report)
    result=dict(schema=schema,virtual_now=virtual_now.isoformat(),hours=hours,
                dashboard=f'http://127.0.0.1:8001/demo?schema={schema}')
    progress(json.dumps(result,indent=2));return result

def replay(settings,hours=3,start=None,progress=print):
    # Freeze the active model set for the entire replay; weekly promotion cannot
    # switch model versions halfway through a test. Temporary copies are removed.
    import tempfile
    import shutil
    from pathlib import Path
    if hours<1 or hours>48: raise ValueError('--hours must be 1..48')
    manifest_path=settings.model_dir/'active.json'
    manifest_bytes=manifest_path.read_bytes()
    manifest=json.loads(manifest_bytes)
    with tempfile.TemporaryDirectory(prefix='demand-monitor-demo-models-') as folder:
        root=Path(folder)
        (root/'active.json').write_bytes(manifest_bytes)
        for mapping in manifest.get('model_files',{}).values():
            for relative in mapping.values():
                source=(settings.model_dir/relative).resolve()
                target=(root/relative).resolve()
                if not source.is_relative_to(settings.model_dir.resolve()) or not target.is_relative_to(root):
                    raise ValueError('Invalid model path')
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
                ref=source.with_suffix('.distribution.json')
                if ref.is_file():shutil.copyfile(ref,target.with_suffix('.distribution.json'))
        return _replay_snapshot(replace(settings,model_dir=root),hours,start,progress)

if __name__=='__main__':
    from .config import get_settings
    p=argparse.ArgumentParser(description='Optional isolated accelerated monitoring replay; no wall-clock sleeps')
    p.add_argument('--hours',type=int,default=3);p.add_argument('--start',help='Timezone-aware origin after fitting period, aligned to hour')
    args=p.parse_args();replay(get_settings(),args.hours,args.start)
