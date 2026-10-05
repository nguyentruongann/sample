"""Optional real PostgreSQL integration; all writes isolated to a random test schema."""
import os,uuid
from dataclasses import replace
import pandas as pd
import pytest
import psycopg
from psycopg import sql
from demand_forecasting.config import get_settings
from demand_forecasting.database import ensure_runtime_tables,upsert_raw_demand,upsert_predictions
from demand_forecasting.monitoring_store import table,load_evaluation,load_windows
from demand_forecasting.performance_monitor import run
from demand_forecasting.monitoring_config import MonitorConfig

@pytest.fixture
def db():
    url=os.getenv('MONITOR_TEST_DATABASE_URL')
    if not url:pytest.skip('Set MONITOR_TEST_DATABASE_URL to an isolated test PostgreSQL')
    settings=replace(get_settings(),database_url=url,postgres_schema='monitor_it_'+uuid.uuid4().hex[:12])
    ensure_runtime_tables(settings)
    try:yield settings
    finally:
        with psycopg.connect(url) as conn:conn.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(settings.postgres_schema)))


def test_archive_exact_labels_and_hourly_retry(db):
    now=pd.Timestamp.now(tz='UTC').tz_convert(db.local_timezone).floor('h')
    t=now-pd.Timedelta(hours=2)
    p=pd.DataFrame([dict(forecast_start_utc=t,hex_id_7='h',travel_mode='Car',horizon_minutes=30,
        model_name='lightgbm',model_version='v1',predicted_demand=50.)])
    upsert_predictions(p,db);p['predicted_demand']=999.;p['model_version']='v2';upsert_predictions(p,db)
    with psycopg.connect(db.database_url) as conn:
        conn.execute(sql.SQL('UPDATE {} SET issued_at=forecast_start_utc').format(table(db,'monitor_prediction_archive')))
    raw=pd.DataFrame([dict(period_datetime_utc=t+pd.Timedelta(minutes=i),hex_id_7='h',travel_mode='Car',total_demand=d,is_holiday=0) for i,d in [(0,10),(10,20),(30,999)]])
    upsert_raw_demand(raw,db)
    f=load_evaluation(db,t,t+pd.Timedelta(hours=1))
    assert f.model_version.iloc[0]=='v1' and f.predicted_demand.iloc[0]==50 and f.actual.iloc[0]==30 and f.bucket_count.iloc[0]==2
    c=MonitorConfig(min_samples=1,min_demand=1)
    run(now,settings=db,config=c)
    first=next(x for x in load_windows(db,as_of=now) if x['travel_mode']=='Car' and x['horizon_minutes']==30)
    assert first['status']=='partial_actual' and first['samples']==0
    fill=raw.iloc[:1].copy();fill['period_datetime_utc']=t+pd.Timedelta(minutes=20);fill['total_demand']=30
    upsert_raw_demand(fill,db);run(now,settings=db,config=c)
    rows=load_windows(db,as_of=now)
    fixed=next(x for x in rows if x['travel_mode']=='Car' and x['horizon_minutes']==30)
    assert fixed['status']=='ready' and fixed['samples']==1 and fixed['actual_sum']==60
    assert len(rows)==6
