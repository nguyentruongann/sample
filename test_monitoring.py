from dataclasses import replace
import json
import numpy as np
import pandas as pd
import pytest
from demand_forecasting.monitoring_config import MonitorConfig
from demand_forecasting.performance_monitor import due_windows,evaluate_window
from demand_forecasting.prediction_reference import make_reference,compare,save_reference,load_reference,model_version,fit_mask
from demand_forecasting.monitoring import prepare,score
from demand_forecasting.feature_monitor import measure,status

NOW=pd.Timestamp('2026-10-05T05:00:00+07:00')
C=MonitorConfig(min_samples=1,min_demand=1)

def predictions():
    t=NOW-pd.Timedelta(hours=2)
    return pd.DataFrame([dict(forecast_start_utc=t+pd.Timedelta(minutes=i*10),hex_id_7='hex',travel_mode='Car',horizon_minutes=60,model_name='lightgbm',model_version='v1',predicted_demand=20.,actual=10.,bucket_count=6,issued_at=t+pd.Timedelta(minutes=i*10,seconds=1)) for i in range(6)])


def test_five_oclock_validates_three_to_four():
    a,b=due_windows(NOW,C)[-1]
    assert a.hour==3 and b.hour==4
    rows=evaluate_window(predictions(),a,b,NOW,C)
    r=next(x for x in rows if x['travel_mode']=='Car' and x['horizon_minutes']==60)
    assert r['samples']==6 and r['actual_sum']==60 and r['prediction_sum']==120
    assert r['wmape_pct']==100 and r['status']=='ready' and len(rows)==6


def test_two_hour_windows_and_cross_midnight():
    a,b=due_windows(pd.Timestamp('2026-10-05T06:00:00+07:00'),replace(C,interval_hours=2))[-1]
    assert a.hour==3 and b.hour==5 and b-a==pd.Timedelta(hours=2)
    a,b=due_windows(pd.Timestamp('2026-10-05T00:00:00+07:00'),C)[-1]
    assert a.day==4 and a.hour==22 and b.hour==23


def test_missing_actual_retried():
    f=predictions();f.loc[0,'bucket_count']=5;a,b=due_windows(NOW,C)[-1]
    r=next(x for x in evaluate_window(f,a,b,NOW,C) if x['travel_mode']=='Car' and x['horizon_minutes']==60)
    assert r['samples']==5 and r['missing_actual']==1 and r['status']=='partial_actual'
    f.loc[0,'bucket_count']=6
    r=next(x for x in evaluate_window(f,a,b,NOW,C) if x['travel_mode']=='Car' and x['horizon_minutes']==60)
    assert r['samples']==6 and r['status']=='ready'


def test_boundary_and_version_separation():
    f=predictions();f.loc[0,'model_version']='v2';extra=f.iloc[:1].copy();extra['forecast_start_utc']=NOW-pd.Timedelta(hours=1)
    a,b=due_windows(NOW,C)[-1]
    rows=[x for x in evaluate_window(pd.concat([f,extra]),a,b,NOW,C) if x['travel_mode']=='Car' and x['horizon_minutes']==60]
    assert len(rows)==2 and sum(x['samples'] for x in rows)==6
    assert all(x['window_forecast_origins']==6 for x in rows)


def test_late_and_maturity():
    f=predictions();f.loc[0,'issued_at']=f.loc[0,'forecast_start_utc']+pd.Timedelta(minutes=5)
    assert prepare(f,NOW).eligible.sum()==5
    assert not prepare(f,NOW-pd.Timedelta(minutes=9)).mature.iloc[-1]


def test_zero_and_weighted_error():
    f=pd.DataFrame({'actual':[1,99],'predicted_demand':[2,89]})
    assert score(f)['wmape_pct']==pytest.approx(11)
    f.actual=0
    assert score(f)['wmape_pct'] is None and score(f)['mae']>0


def test_empty_window():
    a,b=due_windows(NOW,C)[-1];rows=evaluate_window(predictions().iloc[:0],a,b,NOW,C)
    assert len(rows)==6 and all(x['status']=='no_predictions' for x in rows)
    json.dumps(rows,allow_nan=False)


def test_constant_reference_and_range_shift():
    ref=make_reference(np.zeros(100),'v1','Car',10,'a','b')
    assert compare(np.zeros(200),ref)['psi']==pytest.approx(0)
    shifted=compare(np.ones(100)*50,ref)
    assert shifted['psi']>1 and shifted['out_of_reference_range_ratio']==1 and shifted['status']=='warning'
    assert compare([10],ref)['status']=='insufficient_samples'
    json.dumps(shifted,allow_nan=False)


def test_sidecar_preserves_model_and_checks_version(tmp_path):
    path=tmp_path/'model.joblib';path.write_bytes(b'test model')
    r=save_reference(path,[-1,1.6,2.4],'Car',10,'a','b')
    assert path.read_bytes()==b'test model' and r['summary']['min']==0 and r['summary']['max']==2
    assert load_reference(path.with_suffix('.distribution.json'),model_version(path),'Car',10)==r
    with pytest.raises(ValueError):load_reference(path.with_suffix('.distribution.json'),'wrong','Car',10)
    with pytest.raises(ValueError):load_reference(path.with_suffix('.distribution.json'),model_version(path),'Car',60)


def test_reference_excludes_purge_and_test():
    f=pd.DataFrame({'bucket_start':pd.date_range('2026-10-01',periods=24,freq='10min',tz='UTC')})
    a=dict(horizon=60,train_start='2026-10-01T00:00Z',val_start='2026-10-01T02:00Z',test_start='2026-10-01T03:00Z')
    selected=f.loc[fit_mask(f,a),'bucket_start']
    assert len(selected)==8 and selected.max()==pd.Timestamp('2026-10-01T02:00Z')


def test_feature_missing_vs_source_delay():
    f=pd.DataFrame({'hex_id_7':['a','b'],'lag_10m':[1,np.nan],'lag_20m':[2,3]})
    raw=pd.DataFrame({'hex_id_7':['a','b'],'travel_mode':['Car','Car'],'period_datetime_utc':[NOW-pd.Timedelta(minutes=10)]*2})
    s=measure(f,['lag_10m','lag_20m'],raw,NOW,'Car',10,'v',now=NOW)
    assert status(s,NOW,C)['reasons']==['feature_missing_above_threshold']
    later=status(s,NOW+pd.Timedelta(minutes=16),C)
    assert 'source_stale_or_missing' in later['reasons'] and 'feature_snapshot_stale' in later['reasons']


def test_distribution_does_not_need_actual(tmp_path):
    from demand_forecasting.monitoring_service import distribution_report
    serving=predictions().drop(columns=['actual','bucket_count'])
    assert distribution_report(serving,tmp_path,C)[0]['status']=='reference_missing'
    folder=tmp_path/'runs/run1';folder.mkdir(parents=True)
    (folder/'car.distribution.json').write_text(json.dumps(make_reference([20]*100,'v1','Car',60,'a','b')))
    r=distribution_report(serving,tmp_path,C)[0]
    assert r['psi']==0 and r['status']=='ok'


def test_dashboard_asset_and_demo_access_guard():
    from fastapi.testclient import TestClient
    from demand_forecasting.monitoring_service import app
    client=TestClient(app)
    assert 'Phân phối dự đoán' in client.get('/').text
    assert client.get('/demo/report?schema=public').status_code==422
    assert client.get('/demo/report?schema=x%3BDROP').status_code==422


def test_invalid_config(monkeypatch):
    monkeypatch.setenv('MONITOR_INTERVAL_HOURS','3')
    with pytest.raises(ValueError):MonitorConfig.from_env()


def test_replay_isolation_and_no_future_features(tmp_path,monkeypatch):
    import joblib,psycopg
    from test_direct_lightgbm import small_settings
    from demand_forecasting import database,predict,incremental_data,performance_monitor,monitoring_service,monitoring_store
    from demand_forecasting.monitoring_demo import replay
    settings=replace(small_settings(tmp_path),postgres_schema='production',prediction_history_days=1,database_url='postgresql://test:unused@localhost/test')
    settings.model_dir.mkdir(parents=True);mapping={}
    for mode in ('Car','Motorcycle'):
        mapping[mode]={}
        for h in (10,30,60):
            name=f'{mode}_{h}.joblib';joblib.dump({'test_start':'2026-09-01T00:00Z'},settings.model_dir/name);mapping[mode][str(h)]=name
    (settings.model_dir/'active.json').write_text(json.dumps({'model_files':mapping}))
    before={p.name:p.read_bytes() for p in settings.model_dir.iterdir()};seen=[];latest=[None]
    def insert(frame,s):
        assert s.postgres_schema.startswith('production_monitor_demo_');latest[0]=max(frame.period_datetime_utc)
    def forecast(*,forecast_start_utc,settings):
        assert settings.postgres_schema.startswith('production_monitor_demo_')
        assert latest[0]==forecast_start_utc-pd.Timedelta(minutes=10);seen.append(forecast_start_utc)
    class Conn:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def execute(self,*a):return self
    monkeypatch.setattr(psycopg,'connect',lambda *a,**kw:Conn())
    monkeypatch.setattr(database,'ensure_runtime_tables',lambda s:None)
    monkeypatch.setattr(database,'upsert_raw_demand',insert)
    monkeypatch.setattr(incremental_data,'generate_live_bucket',lambda t:pd.DataFrame({'period_datetime_utc':[t]}))
    monkeypatch.setattr(predict,'predict_and_store',forecast)
    monkeypatch.setattr(monitoring_store,'ensure_tables',lambda *a:None)
    monkeypatch.setattr(monitoring_store,'write_state',lambda *a:None)
    evaluated=[];monkeypatch.setattr(performance_monitor,'run',lambda t,**kw:evaluated.append(t))
    monkeypatch.setattr(monitoring_service,'collect',lambda **kw:{'features':[]})
    result=replay(settings,3,'2026-10-05T03:00:00+07:00',progress=lambda s:None)
    assert len(seen)==18 and len(evaluated)==3 and result['schema']!='production'
    assert result['virtual_now']=='2026-10-05T07:00:00+07:00'
    assert before=={p.name:p.read_bytes() for p in settings.model_dir.iterdir()}
