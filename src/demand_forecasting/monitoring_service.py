"""Three mentor monitors: features, delayed performance, prediction distribution."""
import asyncio
import json
import logging
import time
from pathlib import Path
from contextlib import asynccontextmanager

import pandas as pd
import psycopg
from fastapi import FastAPI,HTTPException
from fastapi.responses import FileResponse,Response
from prometheus_client import CollectorRegistry,Gauge,generate_latest,CONTENT_TYPE_LATEST
from .config import get_settings
from .monitoring_config import MonitorConfig
from .monitoring import serializable
from .monitoring_store import ensure_tables,load_states,load_windows,load_serving
from .feature_monitor import status as feature_status
from .prediction_reference import compare,load_reference,model_version

STATE=dict(ok=False,last_success=0,duration=0,report=None,failures=0)
log=logging.getLogger(__name__)


def distribution_report(serving,model_dir,config):
    references={};invalid=set()
    for path in Path(model_dir).glob('runs/*/*.distribution.json'):
        key=None
        try:
            r=json.loads(path.read_text());key=(r['travel_mode'],int(r['horizon_minutes']),r['model_version'])
            references[key]=load_reference(path,key[2],key[0],key[1])
        except (ValueError,KeyError,TypeError):
            # An invalid reference is surfaced by the group's missing/invalid status.
            if key is not None: invalid.add(key)
    groups={} if serving.empty else {key:part for key,part in serving.groupby(['travel_mode','horizon_minutes','model_version'])}
    active=Path(model_dir)/'active.json'
    if active.is_file():
        manifest=json.loads(active.read_text())
        for mode,mapping in manifest.get('model_files',{}).items():
            for h,relative in mapping.items():
                path=(Path(model_dir)/relative).resolve()
                if path.is_relative_to(Path(model_dir).resolve()) and path.is_file():
                    groups.setdefault((mode,int(h),model_version(path)),serving.iloc[:0])
    rows=[]
    for (mode,h,version),part in sorted(groups.items()):
        key=(mode,int(h),version);ref=references.get(key)
        if not ref:
            result=dict(status='reference_invalid' if key in invalid else 'reference_missing',samples=len(part),
                        reference_samples=0,psi=None,out_of_reference_range_ratio=None)
        else:
            result=compare(part.predicted_demand.to_numpy(float),ref,config.min_samples,config.psi_warning,config.range_warning_ratio)
        rows.append(dict(travel_mode=mode,horizon_minutes=int(h),model_version=version,**result))
    return rows


def collect(settings=None,now=None,config=None):
    settings=settings or get_settings();c=config or MonitorConfig.from_env()
    now=pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    with psycopg.connect(settings.require_database_url(),connect_timeout=5) as conn:
        ensure_tables(conn,settings)
    states=load_states(settings)
    features=[]
    for mode in ('Car','Motorcycle'):
        for h in (10,30,60):
            snapshot=states.get(f'feature:{mode}:{h}')
            features.append(feature_status(snapshot,now,c) if snapshot else
                dict(travel_mode=mode,horizon_minutes=h,model_version='unknown',status='not_available',
                     reasons=['no_inference_snapshot'],missing_by_feature={}))
    windows=load_windows(settings,max(72,c.recheck_hours+c.delay_hours),as_of=now)
    from .performance_monitor import due_windows
    allowed_end=due_windows(now,c,settings.local_timezone)[-1][1]
    windows=[r for r in windows
             if pd.Timestamp(r['window_end'])-pd.Timestamp(r['window_start'])==pd.Timedelta(hours=c.interval_hours)
             and pd.Timestamp(r['window_end'])<=allowed_end]
    latest=max((r['window_end'] for r in windows),default=None)
    performance=[r for r in windows if r['window_end']==latest]
    start=now-pd.Timedelta(hours=c.serving_window_hours)
    distribution=distribution_report(load_serving(settings,start,now),settings.model_dir,c)
    return serializable(dict(generated_at=now.isoformat(),features=features,performance=performance,
        performance_history=windows,performance_job=states.get('performance_job'),
        distribution=distribution,serving_start=start.isoformat(),serving_end=now.isoformat(),
        config=c.__dict__))


async def poll():
    interval=MonitorConfig.from_env().poll_seconds
    while True:
        started=time.perf_counter()
        try:
            result=await asyncio.to_thread(collect)
            STATE.update(ok=True,last_success=time.time(),report=result)
        except Exception as error:
            STATE['ok']=False;STATE['failures']+=1
            log.error('Monitoring collection failed (%s); check DB, reference files and row budget',type(error).__name__)
        STATE['duration']=time.perf_counter()-started
        await asyncio.sleep(interval)


@asynccontextmanager
async def lifespan(app):
    task=asyncio.create_task(poll())
    yield
    task.cancel()
    try: await task
    except asyncio.CancelledError: pass


app=FastAPI(title='Demand monitoring',lifespan=lifespan)

@app.get('/')
def dashboard():
    path=Path(__file__).parent/'dashboard_assets/monitoring.html'
    if not path.is_file(): raise HTTPException(503,'monitoring.html missing; rebuild monitor image with packaged dashboard assets')
    return FileResponse(path)

@app.get('/health')
def health():
    if not STATE['ok']: raise HTTPException(503,'Monitoring collector not ready')
    return {'status':'ok'}

@app.get('/report')
def snapshot():
    return dict(collector_ok=STATE['ok'],last_success=STATE['last_success'],failures=STATE['failures'],report=STATE['report'])


def metric_payload(state):
    reg=CollectorRegistry();gauges={}
    def emit(name,value,labels=None):
        if value is None:return
        labels=labels or {}
        if name not in gauges: gauges[name]=Gauge('demand_'+name,name,list(labels),registry=reg)
        g=gauges[name];(g.labels(**{k:str(v) for k,v in labels.items()}) if labels else g).set(value)
    emit('monitor_success',int(state['ok']));emit('monitor_last_success_timestamp',state['last_success'])
    emit('monitor_duration_seconds',state['duration']);emit('monitor_failures',state['failures'])
    r=state['report']
    if r:
        for row in r['features']:
            labels={k:row[k] for k in ['travel_mode','horizon_minutes']}
            emit('feature_warning',int(row['status']!='ok'),labels)
            for key in ['source_age_p95_seconds','source_age_max_seconds','snapshot_age_seconds','hexes_without_history']:
                emit(key,row.get(key),labels)
            for name,ratio in row['missing_by_feature'].items():
                emit('feature_missing_ratio',ratio,dict(labels,feature=name))
        for row in r['performance']:
            labels={k:row[k] for k in ['travel_mode','horizon_minutes','model_version']}
            for key in ['wmape_pct','mae','bias_pct','samples','actual_sum','actual_coverage_pct','pending','missing_actual','late','forecasts','forecast_origins','window_forecast_origins','expected_origins','alert_ready']:
                emit('performance_'+key,row.get(key),labels)
            emit('performance_window_end_timestamp',pd.Timestamp(row['window_end']).timestamp(),labels)
        job=r.get('performance_job')
        emit('performance_last_success_timestamp',pd.Timestamp(job['updated_at']).timestamp() if job else 0)
        emit('performance_interval_seconds',r['config']['interval_hours']*3600)
        for row in r['distribution']:
            labels={k:row[k] for k in ['travel_mode','horizon_minutes','model_version']}
            emit('distribution_warning',int(row['status']=='warning'),labels)
            emit('distribution_reference_missing',int(row['status'] in ('reference_missing','reference_invalid')),labels)
            for key in ['psi','samples','reference_samples','out_of_reference_range_ratio']:
                emit('distribution_'+key,row.get(key),labels)
    return generate_latest(reg)

@app.get('/metrics')
def metrics():
    return Response(metric_payload(STATE),headers={'Content-Type':CONTENT_TYPE_LATEST})


@app.get('/demo')
def demo_dashboard():
    return dashboard()

@app.get('/demo/report')
def demo_snapshot(schema:str):
    from dataclasses import replace
    from .monitoring_demo import validate_demo_schema
    from .monitoring_store import read_state
    settings=get_settings()
    try:validate_demo_schema(settings,schema)
    except ValueError:raise HTTPException(422,'Invalid demo schema') from None
    try:result=read_state(replace(settings,postgres_schema=schema),'demo_report')
    except Exception:raise HTTPException(404,'Demo results not available; replay may still be running') from None
    if result is None:raise HTTPException(404,'Demo replay has not finished')
    return dict(collector_ok=True,last_success=pd.Timestamp(result['updated_at']).timestamp(),failures=0,report=result)
