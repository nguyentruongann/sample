"""Versioned distribution of final-model predictions on the exact TRAIN+VAL fit rows.

Sidecars never modify model bytes/version or influence training/promotion.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
import numpy as np
import pandas as pd

SCHEMA='prediction_reference_v1'

def model_version(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()[:16]

def processed(values):
    x=np.asarray(values,dtype=float)
    if not np.isfinite(x).all(): raise ValueError('Nonfinite reference prediction')
    return np.rint(np.clip(x,0,None))

def histogram(values,cuts):
    return np.bincount(np.searchsorted(cuts,np.asarray(values,float),side='right'),minlength=len(cuts)+1)

def summary(values):
    x=np.asarray(values,float)
    if not len(x): return dict(min=None,max=None,p50=None,p90=None,p95=None)
    return dict(min=float(x.min()),max=float(x.max()),p50=float(np.quantile(x,.5)),
                p90=float(np.quantile(x,.9)),p95=float(np.quantile(x,.95)))

def make_reference(values,version,mode,horizon,start,end):
    x=processed(values)
    if not len(x): raise ValueError('Empty fit reference')
    cuts=np.unique(np.r_[x.min(),np.quantile(x,np.arange(.1,1,.1)),x.max()+.5])
    return dict(schema=SCHEMA,model_version=version,travel_mode=mode,horizon_minutes=int(horizon),
                source='final_model_predictions_on_train_plus_validation_purged',
                postprocessing='clip_rint_only',fit_start=str(start),fit_end_exclusive=str(end),
                created_at=pd.Timestamp.now(tz='UTC').isoformat(),samples=len(x),
                cuts=cuts.tolist(),counts=histogram(x,cuts).tolist(),summary=summary(x))

def save_reference(model_path,values,mode,horizon,start,end):
    model_path=Path(model_path)
    ref=make_reference(values,model_version(model_path),mode,horizon,start,end)
    path=model_path.with_suffix('.distribution.json')
    tmp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        tmp.write_text(json.dumps(ref,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        os.replace(tmp,path)
    finally: tmp.unlink(missing_ok=True)
    return ref

def load_reference(path,version,mode,horizon):
    path=Path(path)
    if not path.is_file(): return None
    r=json.loads(path.read_text(encoding='utf-8'))
    if (r.get('schema')!=SCHEMA or r.get('model_version')!=version
        or r.get('travel_mode')!=mode or r.get('horizon_minutes')!=int(horizon)
        or r.get('postprocessing')!='clip_rint_only'):
        raise ValueError('Reference does not match model version/mode/horizon')
    cuts=np.asarray(r['cuts'],float);counts=np.asarray(r['counts'],float)
    if (not np.isfinite(cuts).all() or not np.all(np.diff(cuts)>0)
        or len(counts)!=len(cuts)+1 or not np.isfinite(counts).all()
        or (counts<0).any() or counts.sum()!=r['samples'] or r['samples']<=0):
        raise ValueError('Invalid reference histogram')
    return r

def compare(values,reference,min_samples=100,psi_warning=.2,range_warning_ratio=.05):
    x=np.asarray(values,float)
    if not np.isfinite(x).all(): raise ValueError('Nonfinite serving predictions')
    counts=histogram(x,reference['cuts'])
    # Equal-proportion smoothing; independent of reference sample size.
    a=np.asarray(reference['counts'],float)/reference['samples']
    b=counts/max(len(x),1)
    aa=np.maximum(a,1e-6);aa/=aa.sum();bb=np.maximum(b,1e-6);bb/=bb.sum()
    value=float(((bb-aa)*np.log(bb/aa)).sum()) if len(x) else None
    out=float(((x<reference['summary']['min'])|(x>reference['summary']['max'])).mean()) if len(x) else None
    enough=len(x)>=min_samples and reference['samples']>=min_samples
    status=('insufficient_samples' if not enough else
            'warning' if value>psi_warning or out>range_warning_ratio else 'ok')
    return dict(samples=len(x),reference_samples=reference['samples'],psi=value,
                out_of_reference_range_ratio=out,status=status,reference=reference['summary'],
                serving=summary(x),cuts=reference['cuts'],reference_ratio=a.tolist(),serving_ratio=b.tolist(),
                source=reference['source'])

def fit_mask(frame,artifact):
    t=pd.to_datetime(frame.bucket_start);h=pd.Timedelta(minutes=int(artifact['horizon']))
    start=pd.Timestamp(artifact['train_start']);val=pd.Timestamp(artifact['val_start']);end=pd.Timestamp(artifact['test_start'])
    return ((t>=start)&(t<val)&(t+h<=val))|((t>=val)&(t<end)&(t+h<=end))

def predict_fit(model,frame,features,horizon,batch_size=100000):
    from .train import _model_input
    output=[]
    for start in range(0,len(frame),batch_size):
        output.append(processed(model.predict(_model_input(frame.iloc[start:start+batch_size],features,horizon))))
    if not output: raise ValueError('No fitting rows available')
    return np.concatenate(output)

def backfill(settings,feature_path):
    """Generate references for existing active model files without retraining.

    Require matching run metrics/fit row count and the original saved feature
    snapshot. Never fall back to recent serving data or test predictions.
    """
    import joblib
    from .feature_engineering import prepare_horizon_frame
    manifest=json.loads((settings.model_dir/'active.json').read_text())
    raw=pd.read_parquet(feature_path);written=[]
    for mode,mapping in manifest['model_files'].items():
        source=raw.loc[raw.travel_mode==mode].copy().reset_index(drop=True)
        for horizon,relative in mapping.items():
            path=(settings.model_dir/relative).resolve()
            if not path.is_relative_to(settings.model_dir.resolve()): raise ValueError('Invalid model path')
            version=model_version(path);sidecar=path.with_suffix('.distribution.json')
            if load_reference(sidecar,version,mode,int(horizon)) is not None: continue
            artifact=joblib.load(path);frame,features,_=prepare_horizon_frame(source,int(horizon))
            if features!=artifact['feature_columns']: raise ValueError('Saved feature schema mismatch')
            rows=frame.loc[fit_mask(frame,artifact)]
            metrics=pd.read_csv(path.parent/'metrics.csv')
            match=metrics.loc[(metrics.travel_mode==mode)&(metrics.horizon==int(horizon))]
            if len(match)!=1 or int(match.iloc[0].fit_rows)!=len(rows):
                raise ValueError('Feature snapshot does not reproduce original fit row count; use original snapshot or retrain')
            # Check mapping as well; do not silently infer a different hex vocabulary.
            mapping_actual={str(hx):int(code) for hx,code in frame[['hex_id_7','hex_code']].drop_duplicates().itertuples(index=False,name=None)}
            if mapping_actual!=artifact['hex_code_mapping']: raise ValueError('Hex mapping differs from training')
            values=predict_fit(artifact['model'],rows,features,int(horizon))
            save_reference(path,values,mode,int(horizon),artifact['train_start'],artifact['test_start'])
            written.append(str(sidecar))
    return written

if __name__=='__main__':
    from .config import get_settings
    parser=argparse.ArgumentParser(description='Build missing distribution references for existing active models')
    parser.add_argument('--feature-path',help='Original saved training feature parquet')
    args=parser.parse_args();settings=get_settings()
    feature_path=Path(args.feature_path) if args.feature_path else settings.data_dir/'processed/demand_features.parquet'
    print(json.dumps(backfill(settings,feature_path),indent=2))
