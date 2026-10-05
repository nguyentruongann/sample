"""Pure monitoring calculations; no training, calibration or prediction mutation."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

def serializable(value):
    return json.loads(json.dumps(value, default=lambda x: x.item() if isinstance(x,np.generic) else str(x), allow_nan=False))


KEYS=['travel_mode','horizon_minutes','model_version']


def prepare(frame, now, wait_seconds=120, issue_grace_seconds=120):
    f=frame.copy()
    if f.empty:
        for c in ['mature','complete','on_time','eligible']: f[c]=pd.Series(dtype=bool)
        return f
    f['forecast_start_utc']=pd.to_datetime(f.forecast_start_utc,utc=True)
    f['issued_at']=pd.to_datetime(f.issued_at,utc=True)
    end=f.forecast_start_utc+pd.to_timedelta(f.horizon_minutes,unit='min')
    f['mature']=end+pd.Timedelta(seconds=wait_seconds)<=pd.Timestamp(now)
    f['complete']=(f.bucket_count==f.horizon_minutes//10)&f.actual.notna()
    f['on_time']=(f.issued_at<=f.forecast_start_utc+pd.Timedelta(seconds=issue_grace_seconds))&(f.issued_at<end)
    f['eligible']=f.mature&f.complete&f.on_time
    return f


def score(f):
    actual=f.actual.to_numpy(float); pred=f.predicted_demand.to_numpy(float)
    total=float(actual.sum()); n=len(f)
    return dict(samples=n, actual_sum=total,
                wmape_pct=float(abs(pred-actual).sum()/total*100) if total>0 else None,
                mae=float(abs(pred-actual).mean()) if n else None,
                bias_pct=float((pred-actual).sum()/total*100) if total>0 else None)

