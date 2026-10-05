"""Missing-feature ratios and source freshness captured before model.predict."""
import logging
import numpy as np
import pandas as pd
from .monitoring_config import MonitorConfig


def measure(frame,features,raw,forecast_start,mode,horizon,version,now=None):
    now=pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    source=raw.loc[(raw.travel_mode==mode)&raw.hex_id_7.astype(str).isin(frame.hex_id_7.astype(str))].copy()
    source['period_datetime_utc']=pd.to_datetime(source.period_datetime_utc,utc=True)
    source=source.loc[source.period_datetime_utc<pd.Timestamp(forecast_start)]
    latest=source.groupby('hex_id_7').period_datetime_utc.max().reindex(frame.hex_id_7.astype(str))
    close=latest+pd.Timedelta(minutes=10)
    timestamps=[t.timestamp() for t in close.dropna()]
    missing={name:float((~np.isfinite(frame[name].to_numpy(float))).mean()) for name in features}
    return dict(travel_mode=mode,horizon_minutes=int(horizon),model_version=version,
        forecast_start_utc=pd.Timestamp(forecast_start).isoformat(),observed_at=now.isoformat(),
        expected_hexes=len(frame),missing_by_feature=missing,
        missing_feature_fraction=float(np.mean(list(missing.values()))),
        hexes_without_history=int(close.isna().sum()),
        source_close_p05_timestamp=float(np.quantile(timestamps,.05)) if timestamps else None,
        source_close_min_timestamp=float(min(timestamps)) if timestamps else None)


def status(snapshot,now,config=None):
    c=config or MonitorConfig.from_env();result=dict(snapshot)
    now=pd.Timestamp(now)
    p05=snapshot.get('source_close_p05_timestamp');oldest=snapshot.get('source_close_min_timestamp')
    age=max(0,now.timestamp()-p05) if p05 is not None else None
    worst=max(0,now.timestamp()-oldest) if oldest is not None else None
    snapshot_age=max(0,(now-pd.Timestamp(snapshot['observed_at'])).total_seconds())
    reasons=[]
    if age is None or worst>c.feature_delay_warning_seconds: reasons.append('source_stale_or_missing')
    if snapshot_age>c.feature_delay_warning_seconds: reasons.append('feature_snapshot_stale')
    if snapshot.get('hexes_without_history',0): reasons.append('hex_history_missing')
    if any(v>c.feature_missing_warning_ratio for v in snapshot['missing_by_feature'].values()):
        reasons.append('feature_missing_above_threshold')
    result.update(source_age_p95_seconds=age,source_age_max_seconds=worst,
                  snapshot_age_seconds=snapshot_age,status='warning' if reasons else 'ok',reasons=reasons)
    return result


def persist(settings,snapshot):
    # Instrument failures without masking the actual inference exception.
    try:
        import psycopg
        from .monitoring_store import ensure_tables,write_state
        with psycopg.connect(settings.require_database_url(),connect_timeout=5) as conn:
            conn.execute("SET statement_timeout='5s'");ensure_tables(conn,settings)
            write_state(conn,settings,f"feature:{snapshot['travel_mode']}:{snapshot['horizon_minutes']}",snapshot)
    except Exception:
        logging.getLogger(__name__).warning('Feature monitoring could not persist snapshot')
