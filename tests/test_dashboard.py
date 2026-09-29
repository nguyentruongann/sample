from datetime import datetime, timezone

import h3
from fastapi.testclient import TestClient

import demand_forecasting.api as api
import demand_forecasting.dashboard as dashboard
from demand_forecasting.config import Settings


CELL = h3.latlng_to_cell(10.7769, 106.7009, 7)
START = datetime(2026, 9, 26, 14, 30, tzinfo=timezone.utc)
OBSERVED = datetime(2026, 9, 26, 14, 20, tzinfo=timezone.utc)


def test_dashboard_routes_keep_forecast_and_observation_separate(monkeypatch):
    monkeypatch.setattr(api, "ensure_runtime_tables", lambda _settings: None)
    received = []

    def snapshot(_settings, mode, horizon, when):
        received.append((mode, horizon, when))
        return {"forecast_start_utc": START.isoformat(), "observed_at_utc": OBSERVED.isoformat()}

    monkeypatch.setattr(api, "load_dashboard_snapshot", snapshot)
    monkeypatch.setattr(api, "load_hex_detail", lambda *_args: {"horizons": {"10": 5, "30": 15}})

    with TestClient(api.app) as client:
        page = client.get("/dashboard")
        assert page.status_code == 200
        assert "DỮ LIỆU GIẢ LẬP" in page.text
        assert client.get("/dashboard/assets/app.js").status_code == 200
        response = client.get("/v1/dashboard/snapshot", params={
            "travel_mode": "Motorcycle", "horizon_minutes": 30,
            "forecast_start_utc": START.isoformat(),
        })
        detail = client.get(f"/v1/dashboard/hex/{CELL}", params={
            "travel_mode": "Car", "forecast_start_utc": START.isoformat(),
        })
        invalid = client.get("/v1/dashboard/hex/invalid", params={"forecast_start_utc": START.isoformat()})
        naive = client.get("/v1/dashboard/snapshot", params={"forecast_start_utc": "2026-09-26T14:30:00"})
        bad_horizon = client.get("/v1/dashboard/snapshot", params={"horizon_minutes": 20})

    assert response.status_code == 200
    assert received == [("Motorcycle", 30, START)]
    assert detail.json()["horizons"] == {"10": 5, "30": 15}
    assert invalid.status_code == naive.status_code == bad_horizon.status_code == 422


def test_snapshot_scopes_batch_and_maps_real_h3_geometry(monkeypatch):
    calls = []
    invalid = "not-an-h3-cell"

    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def execute(self, query, params=None): calls.append((query, params))
        def fetchone(self): return {2: (START,), 5: (OBSERVED,)}[len(calls)]
        def fetchall(self):
            return {
                3: [(CELL, 12.5, START, "version1"), (invalid, 20.0, START, "version1")],
                4: [(START,)],
                6: [(CELL, 11)],
                7: [(OBSERVED, 11)],
            }[len(calls)]

    class Connection:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def cursor(self): return Cursor()

    monkeypatch.setattr(dashboard.psycopg, "connect", lambda *_args, **_kwargs: Connection())
    settings = Settings(
        database_url="postgresql://example.invalid/db", postgres_schema="public",
        postgres_table="fake_demand_10min", predictions_table="demand_predictions",
        seed=1, local_timezone="Asia/Ho_Chi_Minh", train_ratio=.8,
        max_train_rows=None, model_names=("lightgbm",), horizons=(10, 30, 60),
        validation_days=30, test_start=START.isoformat(), data_dir=None,
        model_dir=None, output_dir=None, prediction_history_days=8,
    )
    result = dashboard.load_dashboard_snapshot(settings, "Car", 30)

    assert calls[1][1] == ("Car", 30)
    assert calls[2][1] == (START, "Car", 30)
    assert result["forecast_start_utc"] == START.isoformat()
    assert result["observed_at_utc"] == OBSERVED.isoformat()
    assert result["unmapped_cells"] == 1
    assert result["observed_total"] == 11
    assert result["cells"][0]["predicted_demand"] == 12.5
    assert result["cells"][0]["observed_demand"] == 11
    assert len(result["cells"][0]["boundary"]) == 6
    assert abs(result["cells"][0]["center"][0] - 106.7009) < .01
