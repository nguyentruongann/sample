"""Deterministic fake demand for one live 10-minute bucket."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache

import holidays
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from .config import Settings, get_settings
from .fake_data import (
    COMMON_HEX, CONTRACT, LOCAL_END, LOCAL_START, N_DAYS, N_TIME, SPATIAL_LOG_WEIGHT,
    VALID_MODES, ZERO_RATE_BY_MODE, calibrated_segments, select_bucket_positions,
    time_profile,
)

LIVE_GENERATOR_VERSION = "notebook-smoothed-v4"
HEX_INDEX = np.arange(len(COMMON_HEX))
SLOTS_PER_DAY = 144


@lru_cache(maxsize=2)
def _calibrated_cdf(mode: str) -> np.ndarray:
    """Notebook's calibrated positive demand marks (zero placement is separate)."""
    counts, supports, probabilities = calibrated_segments(mode)
    mass = np.zeros(int(CONTRACT[mode]["max"]) + 1, dtype=np.float64)
    for count, support, probability in zip(counts, supports, probabilities):
        mass[support] += float(count) * probability / CONTRACT[mode]["n"]
    cdf = np.cumsum(mass)
    cdf[-1] = 1.0
    return cdf


@lru_cache(maxsize=2)
def _observed_days(mode: str) -> np.ndarray:
    """Replay the notebook's seeded, mode-specific H3 observation calendar."""
    rng = np.random.default_rng(20260916 + 10_000 * (VALID_MODES.index(mode) + 1))
    hex_index, time_index = select_bucket_positions(CONTRACT[mode]["n"], rng, mode)
    observed = np.zeros((N_DAYS, len(COMMON_HEX)), dtype=bool)
    observed[time_index // 144, hex_index] = True
    return observed


def active_hex_indices(mode: str, local_timestamp: pd.Timestamp) -> np.ndarray:
    """Cycle the calibrated observation-day pattern beyond the notebook range."""
    day = (local_timestamp.normalize() - LOCAL_START.normalize()).days % N_DAYS
    return HEX_INDEX[_observed_days(mode)[day]]


def live_expected_rows(bucket_start_utc, settings: Settings | None = None) -> int:
    settings = settings or get_settings()
    local = normalize_bucket_start(bucket_start_utc).tz_convert(settings.local_timezone)
    return sum(len(active_hex_indices(mode, local)) for mode in VALID_MODES)


@lru_cache(maxsize=2)
def _reference_data(mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Weekly H3/time propensity ranks used by the notebook's mark assignment."""
    time_index = np.repeat(np.arange(7 * 144), len(COMMON_HEX))
    hex_index = np.tile(HEX_INDEX, 7 * 144)
    spatial = SPATIAL_LOG_WEIGHT + np.random.default_rng(
        20260916 + 10_000 * (VALID_MODES.index(mode) + 1)
    ).normal(0, 0.09, size=len(COMMON_HEX))
    profile = time_profile(mode, time_index, hex_index)
    reference = 1.15 * spatial[hex_index] + 1.15 * np.log(profile + 1e-6)
    observed = _observed_days(mode)[:7].repeat(144, axis=0).ravel()
    slot = time_index[observed] % 144
    overnight = (slot < 30) | (slot >= 138)
    # A global rank-to-mark mapping like the notebook must include its
    # smoothed per-hex variation; otherwise live demand changes much more
    # abruptly than the historical rows despite matching marginal quantiles.
    sampled_noise = np.random.default_rng(20260916 + 30_000 *
                                          (VALID_MODES.index(mode) + 1)).normal(size=observed.sum())
    return reference[observed] + 0.05 + 0.44 * sampled_noise, overnight


@lru_cache(maxsize=2)
def _reference_propensity(mode: str) -> np.ndarray:
    return np.sort(_reference_data(mode)[0])


@lru_cache(maxsize=2)
def _zero_probabilities(mode: str) -> tuple[float, float]:
    reference, overnight = _reference_data(mode)
    low = reference < np.quantile(reference, .26)
    night_share = np.mean(low & overnight)
    day_share = np.mean(low & ~overnight)
    rate = ZERO_RATE_BY_MODE[mode]
    return 2 * rate * .85 / night_share, 2 * rate * .15 / day_share


@lru_cache(maxsize=16)
def _white_noise_day(seed: int, mode_index: int, day: int) -> np.ndarray:
    daily_seed = (seed + day * 65_537 + mode_index * 100_003) % (2**32)
    return np.random.default_rng(daily_seed).normal(
        size=(len(COMMON_HEX), SLOTS_PER_DAY)
    ).astype(np.float32)


@lru_cache(maxsize=8)
def _smooth_day(seed: int, mode_index: int, day: int) -> np.ndarray:
    """80-minute correlated per-H3 variation, continuous at midnight."""
    padding = 32  # 4 sigma either side, as in gaussian_filter1d's default.
    white = np.concatenate([
        _white_noise_day(seed, mode_index, day - 1)[:, -padding:],
        _white_noise_day(seed, mode_index, day),
        _white_noise_day(seed, mode_index, day + 1)[:, :padding],
    ], axis=1)
    smoothed = gaussian_filter1d(white, sigma=8.0, axis=1)[:, padding:-padding]
    # Std of Gaussian-filtered unit white noise, matching the notebook's
    # per-H3 normalization over its four-month history.
    kernel = np.exp(-0.5 * (np.arange(-32, 33) / 8.0) ** 2)
    kernel /= kernel.sum()
    return smoothed / np.sqrt(np.square(kernel).sum())


@dataclass(frozen=True)
class LiveBucketSummary:
    bucket_start_utc: str
    rows: int
    total_demand: int
    inserted_rows: int

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


def normalize_bucket_start(value=None) -> pd.Timestamp:
    timestamp = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.floor("10min")


def generate_live_bucket(
    bucket_start_utc=None,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Generate every H3/mode row for one bucket; reruns are deterministic."""

    settings = settings or get_settings()
    bucket_start = normalize_bucket_start(bucket_start_utc)
    local_timestamp = bucket_start.tz_convert(settings.local_timezone)
    vn_holidays = holidays.country_holidays("VN", years=[local_timestamp.year], observed=True)
    is_holiday = int(local_timestamp.date() in vn_holidays)
    bucket_number = int(bucket_start.value // pd.Timedelta(minutes=10).value)

    pieces: list[pd.DataFrame] = []
    for mode_index, mode in enumerate(VALID_MODES):
        mode_seed = (settings.seed + bucket_number * 17 + mode_index * 10_007) % (2**32)
        rng = np.random.default_rng(mode_seed)
        # Preserve the notebook's demand mark distribution and assign higher
        # ranks to its H3/zone/time propensity, with smooth 80-minute noise.
        time_index = int((local_timestamp - LOCAL_START) / pd.Timedelta(minutes=10))
        active = active_hex_indices(mode, local_timestamp)
        profile = time_profile(mode, np.full(len(active), time_index, dtype=np.int64), active)
        spatial = SPATIAL_LOG_WEIGHT + np.random.default_rng(
            20260916 + 10_000 * (mode_index + 1)
        ).normal(0, 0.09, size=len(COMMON_HEX))
        propensity = 1.15 * spatial[active] + 1.15 * np.log(profile + 1e-6)
        if is_holiday:
            hour = local_timestamp.hour + local_timestamp.minute / 60.0
            propensity += 0.06 * (0.30 + 0.25 * np.exp(-0.5 * ((hour - 12) / 2.7) ** 2)
                                  + 0.45 * np.exp(-0.5 * ((hour - 18.5) / 3) ** 2))
        propensity += 0.06 * min(time_index, N_TIME - 1) / max(N_TIME - 1, 1)
        day, slot = divmod(time_index, SLOTS_PER_DAY)
        propensity += (0.44 * _smooth_day(settings.seed, mode_index, day)[active, slot]
                       + 0.08 * rng.normal(size=len(active)))
        reference = _reference_propensity(mode)
        rank = np.searchsorted(reference, propensity, side="left")
        quantile = np.clip((rank + 0.5) / len(reference), 1e-5, 1 - 1e-5)
        demand = np.searchsorted(_calibrated_cdf(mode), quantile, side="left").astype(np.int16)
        # The notebook places about 85% of rare zeros overnight in low
        # propensity cells, never in adjacent ten-minute buckets on one H3.
        slot = local_timestamp.hour * 6 + local_timestamp.minute // 10
        overnight = slot < 30 or slot >= 138
        zero_chance = _zero_probabilities(mode)[0 if overnight else 1]
        zero_mask = ((bucket_number % 2 == 0) & (quantile < .26)
                     & (rng.random(len(active)) < zero_chance))
        demand[zero_mask] = 0

        pieces.append(
            pd.DataFrame(
                {
                    "period_datetime_utc": bucket_start,
                    "hex_id_7": np.asarray(COMMON_HEX, dtype=object)[active],
                    "travel_mode": mode,
                    "total_demand": demand,
                    "is_holiday": np.int8(is_holiday),
                }
            )
        )

    result = pd.concat(pieces, ignore_index=True)
    if result.duplicated(["travel_mode", "hex_id_7", "period_datetime_utc"]).any():
        raise RuntimeError("Live generator produced duplicate primary keys")
    return result


def generate_and_store_bucket(
    bucket_start_utc=None,
    settings: Settings | None = None,
) -> LiveBucketSummary:
    from .database import upsert_raw_demand

    settings = settings or get_settings()
    closed = (normalize_bucket_start() - pd.Timedelta(minutes=10)
              if bucket_start_utc is None else bucket_start_utc)
    frame = generate_live_bucket(closed, settings=settings)
    inserted = upsert_raw_demand(frame, settings=settings)
    bucket = pd.Timestamp(frame["period_datetime_utc"].iloc[0]).tz_convert("UTC")
    return LiveBucketSummary(
        bucket_start_utc=bucket.isoformat(),
        rows=len(frame),
        total_demand=int(frame["total_demand"].sum()),
        inserted_rows=inserted,
    )


def backfill_live_history(
    *,
    end_utc=None,
    settings: Settings | None = None,
    batch_buckets: int = 36,
    full_scan: bool = False,
) -> dict[str, int | str]:
    """Repair missing/partial live buckets up to the last closed 10-minute boundary."""
    from .database import (
        ensure_live_generator_version, latest_observation_timestamp,
        live_bucket_row_counts, upsert_raw_demand,
    )

    settings = settings or get_settings()
    if batch_buckets < 1:
        raise ValueError("batch_buckets must be positive")
    end = normalize_bucket_start(end_utc)
    if full_scan:
        rebuilt = ensure_live_generator_version(
            LIVE_GENERATOR_VERSION, LOCAL_END.tz_convert("UTC"), settings
        )
        if rebuilt:
            print("Live fake generator upgraded: rebuilding synthetic rows after notebook history", flush=True)
    recent_start = end - pd.Timedelta(days=settings.prediction_history_days)
    latest = latest_observation_timestamp(settings)
    live_start = LOCAL_END.tz_convert("UTC")
    if full_scan:
        start = live_start
    else:
        # Examine recent buckets as well as every bucket missed during downtime.
        next_missing = latest + pd.Timedelta(minutes=10) if latest is not None else live_start
        start = max(live_start, min(next_missing, recent_start))
    if start >= end:
        return {"start_utc": start.isoformat(), "end_utc": end.isoformat(),
                "buckets": 0, "upserted_rows": 0}
    counts = live_bucket_row_counts(start, end, settings)
    count = 0
    inserted = 0
    current: list[pd.DataFrame] = []
    for timestamp in pd.date_range(start=start, end=end, freq="10min", inclusive="left"):
        expected_rows = live_expected_rows(timestamp, settings)
        if counts.get(timestamp, 0) == expected_rows:
            continue
        if counts.get(timestamp, 0) > expected_rows:
            raise ValueError(f"Unexpected live bucket row count at {timestamp}: {counts[timestamp]}")
        current.append(generate_live_bucket(timestamp, settings=settings))
        count += 1
        if len(current) >= batch_buckets:
            inserted += upsert_raw_demand(pd.concat(current, ignore_index=True), settings=settings)
            current.clear()
    if current:
        inserted += upsert_raw_demand(pd.concat(current, ignore_index=True), settings=settings)
    return {"start_utc": start.isoformat(), "end_utc": end.isoformat(),
            "buckets": count, "upserted_rows": inserted}
