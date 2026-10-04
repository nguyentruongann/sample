"""Versioned training contract extracted from final.ipynb (saved run).

No post-model calibration: the notebook comparison did not justify enabling it.
"""

FEATURE_SCHEMA = "final_fresh_neighbor_v1"
EVALUATION_SCHEMA = 4
FEATURE_POLICY = {10: "fresh_10_20", 30: "fresh_plus_neighbor", 60: "fresh_plus_neighbor"}
EXTRA_FEATURES = {
    10: ["lag_10m", "lag_20m", "mean_10_60m", "delta_10_30m"],
    30: ["lag_10m", "lag_20m", "mean_10_60m", "delta_10_30m",
         "n10", "n20", "n30", "n60", "neighbor_trend_10_30"],
    60: ["lag_10m", "lag_20m", "mean_10_60m", "delta_10_30m",
         "n10", "n20", "n30", "n60", "neighbor_trend_10_30"],
}
EXPECTED_FEATURE_COUNT = {10: 25, 30: 33, 60: 33}
LEAF_TRIAL_ORDER = {10: (130, 31, 63), 30: (63, 31, 130), 60: (63, 31, 130)}
# (num_leaves, n_estimators), from FINAL_FIT_SUMMARY, not guessed/tuned on TEST.
LOCKED_PARAMS = {
    ("Car", 10): (63, 633),
    ("Car", 30): (130, 763),
    ("Car", 60): (130, 799),
    ("Motorcycle", 10): (63, 762),
    ("Motorcycle", 30): (63, 795),
    ("Motorcycle", 60): (130, 792),
}


def feature_columns(horizon: int, base: list[str]) -> list[str]:
    columns = list(base)
    if horizon > 10:
        columns += [f"naive_recent_h{horizon}", f"naive_1d_h{horizon}", f"naive_7d_h{horizon}"]
    columns += EXTRA_FEATURES[horizon]
    if len(columns) != EXPECTED_FEATURE_COUNT[horizon] or len(set(columns)) != len(columns):
        raise ValueError("Feature contract mismatch")
    return columns
