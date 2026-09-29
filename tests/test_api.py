import pandas as pd
from fastapi.testclient import TestClient

import demand_forecasting.api as api_module


def test_health_and_latest_prediction(monkeypatch):
    monkeypatch.setattr(api_module, "ensure_runtime_tables", lambda _settings: None)
    monkeypatch.setattr(api_module, "database_is_ready", lambda _settings: True)
    monkeypatch.setattr(
        api_module,
        "load_latest_predictions",
        lambda **_kwargs: pd.DataFrame(
            {
                "forecast_start_utc": [pd.Timestamp("2026-09-22T10:10:00Z")],
                "hex_id_7": ["8765b56e9ffffff"],
                "travel_mode": ["Car"],
                "horizon_minutes": [10],
                "model_name": ["lightgbm"],
                "model_version": ["abc123"],
                "predicted_demand": [12.5],
                "created_at": [pd.Timestamp("2026-09-22T10:10:02Z")],
            }
        ),
    )

    with TestClient(api_module.app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        response = client.get(
            "/v1/predictions/latest",
            params={"travel_mode": "Car", "horizon_minutes": 10},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["predictions"][0]["predicted_demand"] == 12.5
