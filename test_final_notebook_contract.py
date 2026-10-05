"""Golden parity against the supplied final.ipynb, including gaps and code maps."""
import gc
import json
from dataclasses import replace
from pathlib import Path

import h3
import holidays
import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from demand_forecasting.feature_engineering import build_features, prepare_horizon_frame
from demand_forecasting.notebook_contract import FEATURE_SCHEMA, LOCKED_PARAMS
from demand_forecasting.predict import build_online_features
from demand_forecasting.data_split import time_based_split
import demand_forecasting.train as training
from test_direct_lightgbm import small_settings, raw_history


def sparse_raw():
    center = h3.latlng_to_cell(10.78, 106.69, 7)
    cells = sorted(h3.grid_disk(center, 1))
    times = pd.date_range("2026-08-20", periods=20*144, freq="10min", tz="Asia/Ho_Chi_Minh")
    frames = []
    for mode in ("Car", "Motorcycle"):
        for j,hx in enumerate(cells):
            # Different mode-specific vocabularies: H10 global codes vs H30/H60 factorization.
            if mode == "Motorcycle" and j == 0:
                continue
            ix = np.arange(len(times))
            keep = ((ix+j)%37 != 0)
            frames.append(pd.DataFrame({"period_datetime_utc":times[keep].tz_convert("UTC"),
                "hex_id_7":hx,"travel_mode":mode,
                "total_demand":((ix[keep]*3+j*7)%51).astype(np.int16),
                "is_holiday":0}))  # Deliberately incorrect Sept 2 flag; notebook recomputes.
    return pd.concat(frames,ignore_index=True)


def notebook_reference(raw):
    times=pd.DatetimeIndex(raw.period_datetime_utc).tz_convert("Asia/Ho_Chi_Minh")
    df=raw.copy();df["bucket_start"]=times
    # Same retained timeline used by the notebook cutoff assertion.
    timeline=pd.DatetimeIndex(times.unique()).sort_values()[3:]
    cutoff=timeline[int(len(timeline)*.80)]
    cases=[(m,h) for m in ("Car","Motorcycle") for h in (10,30,60)]
    scope=dict(df=df,np=np,pd=pd,gc=gc,h3=h3,csr_matrix=csr_matrix,
        VN_HOLIDAYS=holidays.country_holidays("VN",years=[2026],observed=True),
        TEST_START=cutoff,VAL_START=cutoff-pd.Timedelta(days=2),VALIDATION_DAYS=2,
        MODES=("Car","Motorcycle"),CASES=cases,
        FEATURE_POLICY={k:"fresh_10_20" if k[1]==10 else "fresh_plus_neighbor" for k in cases},
        EXPECTED_FEATURE_COUNT={10:25,30:33,60:33},FRESH_BUCKET_IS_CLOSED_AT_T=True,
        display=lambda *_:None)
    fixture=json.loads((Path(__file__).parent/"fixtures/final_notebook_feature_cells.json").read_text())
    for i in (8,9,11):
        exec(compile(fixture[str(i)],f"original_notebook_cell_{i}","exec"),scope)
    return scope


def test_offline_and_online_match_original_notebook_all_six_cases():
    raw=sparse_raw();reference=notebook_reference(raw)
    offline_base=build_features(raw)
    for mode,h in reference["CASES"]:
        source=offline_base.loc[offline_base.travel_mode==mode].reset_index(drop=True)
        offline,columns,target=prepare_horizon_frame(source,h)
        original=reference["CASE_DATA"][(mode,h)]
        assert columns==reference["FEATURE_COLUMNS"][(mode,h)]
        pd.testing.assert_series_equal(offline.bucket_start,original["frame"].bucket_start)
        np.testing.assert_array_equal(offline[target],original["frame"][original["target"]])
        indices=np.arange(len(offline))
        expected=reference["build_input"]((mode,h),indices)
        np.testing.assert_allclose(offline[columns].to_numpy(float),np.asarray(expected,dtype=float),
                                   rtol=1e-6,atol=1e-6,equal_nan=True)
        split=time_based_split(offline,h,boundaries=(reference["VAL_START"],reference["TEST_START"]),
                               historical_end=offline_base.bucket_start.max()+pd.Timedelta(minutes=10))
        for name in ("train_index","val_index","test_index","train_val_index"):
            np.testing.assert_array_equal(getattr(split,name),reference["SPLITS"][(mode,h)][name])
        mapping={str(hx):int(code) for hx,code in offline[["hex_id_7","hex_code"]].drop_duplicates().itertuples(index=False,name=None)}
        artifact=dict(travel_mode=mode,feature_schema=FEATURE_SCHEMA,feature_columns=columns,
            hex_categories=sorted(mapping),hex_code_mapping=mapping,
            neighbor_hex_ids=sorted(source.hex_id_7.astype(str).unique()),
            training_origin=offline_base.attrs["training_origin"])
        t=pd.Timestamp("2026-09-02T12:00:00+07:00")
        online,cols=build_online_features(raw,t,artifact,h)
        rows=offline.loc[offline.bucket_start==t]
        got=online.set_index("hex_id_7").loc[rows.hex_id_7.astype(str),cols]
        np.testing.assert_allclose(got.to_numpy(float),rows[cols].to_numpy(float),
                                   rtol=1e-6,atol=1e-6,equal_nan=True)
        # Future labels cannot affect current features, even if caller passes them.
        mutated=raw.copy();mutated.loc[mutated.period_datetime_utc>=t,"total_demand"]=30000
        changed,_=build_online_features(mutated,t,artifact,h)
        np.testing.assert_allclose(online[cols],changed[cols],equal_nan=True)


def test_locked_six_models_refit_saved_counts_and_integer_reports(tmp_path,monkeypatch):
    settings=replace(small_settings(tmp_path),training_strategy="locked")
    features=build_features(raw_history())
    real_fit=training._fit
    calls=[]
    def capture(model,x,y,horizon):
        calls.append((horizon,model.n_estimators,model.num_leaves,len(x)))
        return real_fit(model,x,y,horizon)
    monkeypatch.setattr(training,"_fit",capture)
    result=training.train_all(features,settings=settings,rolling=True)
    assert len(calls)==6 and result.fit_count.eq(2).all()
    assert result.parameter_source.eq("final_notebook_locked").all()
    for row in result.to_dict("records"):
        assert (row["num_leaves"],row["best_iteration"])==LOCKED_PARAMS[(row["travel_mode"],row["horizon"])]
        artifact=joblib.load(row["model_path"])
        from demand_forecasting.prediction_reference import load_reference, model_version
        model_path=Path(row["model_path"])
        ref=load_reference(model_path.with_suffix(".distribution.json"),model_version(model_path),row["travel_mode"],row["horizon"])
        assert ref["samples"]==row["fit_rows"]
        assert ref["source"]=="final_model_predictions_on_train_plus_validation_purged"
        assert artifact["model"].n_features_in_==({10:25,30:33,60:33}[row["horizon"]])
        pred=pd.read_parquet(row["predictions_path"])
        np.testing.assert_array_equal(pred.prediction,np.rint(np.clip(pred.prediction_float,0,None)))
        assert row["WMAPE_%"]==pytest.approx(100*np.abs(pred.prediction-pred.actual).sum()/pred.actual.sum())
    # Existing-model migration recreates the same reference from original feature parquet.
    from demand_forecasting.prediction_reference import backfill
    feature_path=settings.data_dir/"original_features.parquet"
    features.to_parquet(feature_path,index=False)
    first_path=Path(result.iloc[0]["model_path"]).with_suffix(".distribution.json")
    original_reference=json.loads(first_path.read_text())
    first_path.unlink()
    assert backfill(settings,feature_path)==[str(first_path)]
    restored=json.loads(first_path.read_text())
    assert restored["model_version"]==original_reference["model_version"]
    assert restored["counts"]==original_reference["counts"]
    assert restored["cuts"]==original_reference["cuts"]
    trials=pd.read_csv(settings.output_dir/"model_search_trials.csv")
    assert (trials.search_val_rows==trials.full_val_rows).all()
    assert (trials.search_train_rows>settings.search_train_rows).all()


def test_notebook_historical_train_is_not_truncated_to_90_days():
    frame=pd.DataFrame({"bucket_start":pd.date_range("2026-01-01",periods=220*144,freq="10min",tz="UTC")})
    cutoff=pd.Timestamp("2026-07-01T00:00:00Z")
    split=time_based_split(frame,10,test_start=cutoff,validation_days=30,
        historical_end=frame.bucket_start.max()+pd.Timedelta(minutes=10))
    assert split.train_start==frame.bucket_start.min()
    assert split.train_index[0]==0
