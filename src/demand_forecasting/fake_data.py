"""Full calibrated fake-demand generator from source_main.ipynb (spatial + sparse zeros)."""

from pathlib import Path
from .config import PROJECT_ROOT
import gc

import h3
import holidays
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
try:
    from IPython.display import display
except ImportError:
    def display(frame):
        print(frame.to_string(index=False))

SEED = 20260916
LOCAL_TIMEZONE = "Asia/Ho_Chi_Minh"

LOCAL_START = pd.Timestamp(
    "2026-05-01 00:00:00",
    tz=LOCAL_TIMEZONE,
)
LOCAL_END = pd.Timestamp(
    "2026-09-11 00:00:00",
    tz=LOCAL_TIMEZONE,
)

# Lịch nghỉ lễ Việt Nam. observed=True bao gồm ngày nghỉ bù do thư viện cung cấp.
VN_HOLIDAYS = holidays.country_holidays(
    "VN",
    years=range(LOCAL_START.year, LOCAL_END.year + 1),
    observed=True,
    language="vi",
)


VALID_MODES = ("Car", "Motorcycle")

# Mỗi ngày có 144 bucket 10 phút
SLOTS_PER_DAY = 24 * 6
N_DAYS = (LOCAL_END - LOCAL_START).days
N_TIME = N_DAYS * SLOTS_PER_DAY

ALL_LOCAL_DATES = pd.date_range(
    start=LOCAL_START.normalize(),
    periods=N_DAYS,
    freq="D",
)

IS_HOLIDAY_BY_DAY = np.asarray(
    [int(ts.date() in VN_HOLIDAYS) for ts in ALL_LOCAL_DATES],
    dtype=np.int8,
)

HOLIDAY_NAME_BY_DAY = np.asarray(
    [VN_HOLIDAYS.get(ts.date(), "") for ts in ALL_LOCAL_DATES],
    dtype=object,
)

HOLIDAY_DATES_IN_RANGE = pd.DataFrame({
    "date": ALL_LOCAL_DATES.date,
    "is_holiday": IS_HOLIDAY_BY_DAY,
    "holiday_name": HOLIDAY_NAME_BY_DAY,
})
HOLIDAY_DATES_IN_RANGE = HOLIDAY_DATES_IN_RANGE[
    HOLIDAY_DATES_IN_RANGE["is_holiday"] == 1
].reset_index(drop=True)

HCM_CENTER = (10.7769, 106.7009)
H3_RESOLUTION = 7
H3_K_RING = 8

# Preserve the notebook's existing 217 H3 IDs and spatial footprint. A radius
# around the city center does not certify membership of the HCMC boundary;
# replace COMMON_HEX with a verified operations H3 list if one is available.
HCM_CENTER_HEX = h3.latlng_to_cell(
    *HCM_CENTER,
    H3_RESOLUTION,
)

COMMON_HEX = sorted(
    h3.grid_disk(HCM_CENTER_HEX, H3_K_RING)
)

N_HEX = len(COMMON_HEX)


def km_to(point, lat, lng):
    """Local equirectangular distance (adequate for this ~40 km footprint)."""
    return np.hypot(
        (lat - point[0]) * 111.1,
        (lng - point[1]) * 109.2,
    )


def make_hex_metadata():
    """Geographic priors: these area labels are synthetic, not actual POIs."""
    coords = np.asarray([h3.cell_to_latlng(h) for h in COMMON_HEX])
    lat, lng = coords[:, 0], coords[:, 1]
    anchor = {
        "central": (10.7769, 106.7009),
        "airport": (10.8188, 106.6519),
        "west": (10.7510, 106.6550),
        "south": (10.7295, 106.7219),
        "east": (10.8506, 106.7719),
    }
    distance = {key: km_to(value, lat, lng) for key, value in anchor.items()}
    near = {key: np.exp(-0.5 * (d / 3.8) ** 2) for key, d in distance.items()}
    outer = np.minimum.reduce([distance[key] for key in anchor]) > 6.2
    zone_score = np.column_stack([
        1.40 * near["central"],
        1.20 * near["airport"],
        near["west"],
        near["south"],
        near["east"],
    ])
    zone = np.array(["central", "airport", "west", "south", "east"], dtype=object)[
        zone_score.argmax(axis=1)
    ]
    zone[outer] = "outer"
    # A shared coarse H3 effect keeps neighboring hexes more similar.
    parent = [h3.cell_to_parent(h, 5) for h in COMMON_HEX]
    parent_code = pd.factorize(np.asarray(parent), sort=True)[0]
    rng = np.random.default_rng(SEED + 402)
    coarse = rng.normal(0, 0.18, size=parent_code.max() + 1)
    local = rng.normal(0, 0.075, size=N_HEX)
    spatial_log = (
        0.72 * near["central"]
        + 0.38 * near["airport"]
        + 0.26 * near["west"]
        + 0.24 * near["south"]
        + 0.22 * near["east"]
        - 0.018 * distance["central"]
        - 0.43 * outer
        + coarse[parent_code]
        + local
    )
    return pd.DataFrame({
        "hex_id_7": COMMON_HEX,
        "center_lat": lat,
        "center_lng": lng,
        "synthetic_zone": zone,
        "spatial_log_weight": spatial_log.astype(np.float32),
    })


HEX_METADATA = make_hex_metadata()
ZONE = HEX_METADATA["synthetic_zone"].to_numpy()
SPATIAL_LOG_WEIGHT = HEX_METADATA["spatial_log_weight"].to_numpy()

# Zero chỉ xuất hiện hiếm ở các active hex. Tỷ lệ nhỏ để không phá
# n/mean/std/p50/p95/p99/max được hiệu chỉnh từ thống kê thực.
ZERO_RATE_BY_MODE = {
    "Car": 0.003,         # khoảng 0.30% số bucket
    "Motorcycle": 0.002,  # khoảng 0.20% số bucket
}
ZERO_CANDIDATE_SHARE = 0.26

CONTRACT = {
    "Car": {
        "n": 2_863_718,
        "mean": 9.75,
        "std": 17.37,
        "p50": 4,
        "p95": 40,
        "p99": 86,
        "max": 1_000,
        "bucket_rows": [
            615_688,
            1_525_941,
            618_033,
            104_056,
        ],
        "bucket_volume_pct": [
            0.022,
            0.227,
            0.454,
            0.298,
        ],
    },
    "Motorcycle": {
        "n": 2_540_812,
        "mean": 16.96,
        "std": 29.29,
        "p50": 6,
        "p95": 73,
        "p99": 144,
        "max": 914,
        "bucket_rows": [
            480_508,
            1_069_539,
            770_448,
            220_316,
        ],
        "bucket_volume_pct": [
            0.011,
            0.108,
            0.400,
            0.481,
        ],
    },
}

# ------------------------------------------------------------
# HÀM TẠO PHÂN PHỐI DEMAND ĐÚNG CONTRACT
# ------------------------------------------------------------
def largest_remainder_counts(weights, total):
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    expected = weights * total
    counts = np.floor(expected).astype(np.int64)

    remainder = total - counts.sum()

    if remainder:
        order = np.argsort(
            -(expected - counts),
            kind="stable",
        )
        counts[order[:remainder]] += 1

    return counts


def bucket_counts_for_mode(mode):
    spec = CONTRACT[mode]

    counts = np.asarray(
        spec["bucket_rows"],
        dtype=np.int64,
    ).copy()

    # Motorcycle thiếu một dòng trong thống kê nguồn
    counts[-1] += spec["n"] - counts.sum()

    return counts


def maxent_discrete(
    support,
    feature_functions,
    targets,
):
    support = np.asarray(support, dtype=float)

    features = np.column_stack([
        function(support)
        for function in feature_functions
    ])

    targets = np.asarray(targets, dtype=float)

    def probabilities(theta):
        logits = features @ theta
        logits -= logits.max()

        probability = np.exp(logits)
        return probability / probability.sum()

    result = least_squares(
        lambda theta:
        probabilities(theta) @ features - targets,
        np.zeros(features.shape[1]),
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
        max_nfev=20_000,
    )

    probability = probabilities(result.x)

    error = np.max(
        np.abs(
            probability @ features - targets
        )
    )

    if error > 2e-7:
        raise RuntimeError(
            f"Calibration failed: {error}"
        )

    return probability


def calibrated_segments(mode):
    spec = CONTRACT[mode]

    n = spec["n"]
    mean = float(spec["mean"])
    std = float(spec["std"])
    max_value = int(spec["max"])

    bucket_counts = bucket_counts_for_mode(mode)
    bucket_probability = bucket_counts / n

    exact_one_share = (
        bucket_counts[0] / (n * mean)
    )

    remaining_share = np.asarray(
        spec["bucket_volume_pct"][1:],
        dtype=float,
    )

    remaining_share *= (
        1.0 - exact_one_share
    ) / remaining_share.sum()

    volume_share = np.r_[
        exact_one_share,
        remaining_share,
    ]

    conditional_mean = (
        mean
        * volume_share
        / bucket_probability
    )

    delta = max(0.0002, 3.0 / n)

    supports = [
        np.asarray([1], dtype=np.int16)
    ]
    probabilities = [
        np.asarray([1.0])
    ]

    # Bucket 2–9
    q50 = int(spec["p50"])
    support = np.arange(2, 10)

    probability = maxent_discrete(
        support,
        [
            lambda x: x / 9.0,
            lambda x:
            (x <= q50 - 1).astype(float),
            lambda x:
            (x <= q50).astype(float),
        ],
        [
            conditional_mean[1] / 9.0,
            (
                0.50
                - delta
                - bucket_probability[0]
            ) / bucket_probability[1],
            (
                0.50
                + delta
                - bucket_probability[0]
            ) / bucket_probability[1],
        ],
    )

    supports.append(support.astype(np.int16))
    probabilities.append(probability)

    # Bucket 10–49
    support = np.arange(10, 50)

    functions = [
        lambda x: x / 49.0
    ]
    targets = [
        conditional_mean[2] / 49.0
    ]

    q95 = int(spec["p95"])

    if q95 <= 49:
        functions.extend([
            lambda x:
            (x <= q95 - 1).astype(float),
            lambda x:
            (x <= q95).astype(float),
        ])

        targets.extend([
            (
                0.95
                - delta
                - bucket_probability[:2].sum()
            ) / bucket_probability[2],
            (
                0.95
                + delta
                - bucket_probability[:2].sum()
            ) / bucket_probability[2],
        ])

    probability = maxent_discrete(
        support,
        functions,
        targets,
    )

    supports.append(support.astype(np.int16))
    probabilities.append(probability)

    # Bucket 50+
    population_second_moment = (
        std**2 * (n - 1) / n
        + mean**2
    )

    lower_second_moment = sum(
        bucket_probability[index]
        * float(
            probabilities[index]
            @ supports[index].astype(float) ** 2
        )
        for index in range(3)
    )

    tail_second_moment = (
        population_second_moment
        - lower_second_moment
    ) / bucket_probability[3]

    support = np.arange(
        50,
        max_value + 1,
    )

    functions = [
        lambda x: x / max_value,
        lambda x:
        (x / max_value) ** 2,
    ]

    targets = [
        conditional_mean[3] / max_value,
        tail_second_moment / max_value**2,
    ]

    for percentile, quantile_value in [
        (0.95, int(spec["p95"])),
        (0.99, int(spec["p99"])),
    ]:
        if quantile_value >= 50:
            functions.extend([
                lambda x, q=quantile_value:
                (x <= q - 1).astype(float),
                lambda x, q=quantile_value:
                (x <= q).astype(float),
            ])

            targets.extend([
                (
                    percentile
                    - delta
                    - bucket_probability[:3].sum()
                ) / bucket_probability[3],
                (
                    percentile
                    + delta
                    - bucket_probability[:3].sum()
                ) / bucket_probability[3],
            ])

    probability = maxent_discrete(
        support,
        functions,
        targets,
    )

    supports.append(support.astype(np.int16))
    probabilities.append(probability)

    return (
        bucket_counts,
        supports,
        probabilities,
    )


def repair_integer_moments(values, mode):
    spec = CONTRACT[mode]

    max_value = int(spec["max"])
    q99 = int(spec["p99"])

    counts = np.bincount(
        values.astype(np.int64),
        minlength=max_value + 1,
    ).astype(np.int64)

    support = np.arange(
        max_value + 1,
        dtype=np.int64,
    )

    n = counts.sum()

    target_sum = int(
        np.rint(n * spec["mean"])
    )

    target_sumsq_float = (
        (n - 1) * spec["std"] ** 2
        + target_sum**2 / n
    )

    target_sumsq = int(
        np.rint(target_sumsq_float)
    )

    if (target_sumsq - target_sum) % 2:
        candidates = (
            target_sumsq - 1,
            target_sumsq + 1,
        )

        target_sumsq = min(
            candidates,
            key=lambda x:
            abs(x - target_sumsq_float),
        )

    current_sum = int(counts @ support)
    sum_deficit = target_sum - current_sum

    if sum_deficit > 0:
        for value in range(
            q99 + 1,
            max_value,
        ):
            if sum_deficit == 0:
                break

            move = min(
                sum_deficit,
                int(counts[value]),
            )

            counts[value] -= move
            counts[value + 1] += move
            sum_deficit -= move

    elif sum_deficit < 0:
        remaining = -sum_deficit

        for value in range(
            max_value,
            q99 + 1,
            -1,
        ):
            if remaining == 0:
                break

            movable = int(counts[value])

            if value == max_value:
                movable = max(
                    0,
                    movable - 1,
                )

            move = min(
                remaining,
                movable,
            )

            counts[value] -= move
            counts[value - 1] += move
            remaining -= move

        sum_deficit = -remaining

    if sum_deficit != 0:
        raise RuntimeError(
            f"Cannot repair mean for {mode}"
        )

    current_sumsq = int(
        counts @ (support * support)
    )

    square_deficit = (
        target_sumsq - current_sumsq
    )

    iterations = 0

    while square_deficit > 0:
        iterations += 1

        if iterations > 100_000:
            break

        max_gap = square_deficit // 2 - 1

        if max_gap < 1:
            break

        available_low = (
            np.flatnonzero(
                counts[q99 + 2:max_value] > 0
            )
            + q99
            + 2
        )

        available_high = (
            np.flatnonzero(
                counts[q99 + 1:max_value] > 0
            )
            + q99
            + 1
        )

        best = None

        for low in available_low:
            high_limit = min(
                max_value - 1,
                int(low) + max_gap,
            )

            candidates = available_high[
                (available_high > low)
                & (available_high <= high_limit)
            ]

            if len(candidates):
                high = int(candidates[-1])
                gap = high - int(low)

                if best is None or gap > best[0]:
                    best = (
                        gap,
                        int(low),
                        high,
                    )

        if best is None:
            break

        gap, low, high = best
        delta_per_pair = 2 * gap + 2

        move = min(
            int(counts[low]),
            int(counts[high]),
            square_deficit // delta_per_pair,
        )

        if move <= 0:
            break

        counts[low] -= move
        counts[low - 1] += move
        counts[high] -= move
        counts[high + 1] += move

        square_deficit -= (
            move * delta_per_pair
        )

    while square_deficit < 0:
        iterations += 1

        if iterations > 100_000:
            break

        remaining = -square_deficit
        max_gap = remaining // 2 + 1

        if max_gap < 2:
            break

        available_low = (
            np.flatnonzero(
                counts[q99 + 1:max_value] > 0
            )
            + q99
            + 1
        )

        high_capacity = counts.copy()
        high_capacity[max_value] = max(
            0,
            high_capacity[max_value] - 1,
        )

        available_high = (
            np.flatnonzero(
                high_capacity[
                    q99 + 2:max_value + 1
                ] > 0
            )
            + q99
            + 2
        )

        best = None

        for low in available_low:
            high_limit = min(
                max_value,
                int(low) + max_gap,
            )

            candidates = available_high[
                (available_high >= low + 2)
                & (available_high <= high_limit)
            ]

            if len(candidates):
                high = int(candidates[-1])
                gap = high - int(low)

                if best is None or gap > best[0]:
                    best = (
                        gap,
                        int(low),
                        high,
                    )

        if best is None:
            break

        gap, low, high = best
        delta_per_pair = 2 * gap - 2

        movable_high = int(counts[high])

        if high == max_value:
            movable_high = max(
                0,
                movable_high - 1,
            )

        move = min(
            int(counts[low]),
            movable_high,
            remaining // delta_per_pair,
        )

        if move <= 0:
            break

        counts[low] -= move
        counts[low + 1] += move
        counts[high] -= move
        counts[high - 1] += move

        square_deficit += (
            move * delta_per_pair
        )

    actual_sum = int(counts @ support)
    actual_sumsq = int(
        counts @ (support * support)
    )

    if (
        actual_sum != target_sum
        or actual_sumsq != target_sumsq
    ):
        raise RuntimeError(
            f"Moment repair failed for {mode}"
        )

    return np.repeat(
        support,
        counts,
    ).astype(np.int16)


def create_calibrated_marks(mode, rng):
    spec = CONTRACT[mode]
    n = spec["n"]

    (
        bucket_counts,
        supports,
        probabilities,
    ) = calibrated_segments(mode)

    pieces = []

    for count, support, probability in zip(
        bucket_counts,
        supports,
        probabilities,
    ):
        value_counts = largest_remainder_counts(
            probability,
            int(count),
        )

        pieces.append(
            np.repeat(
                support,
                value_counts,
            ).astype(np.int16)
        )

    values = np.concatenate(pieces)

    max_value = int(spec["max"])

    if not np.any(values == max_value):
        tail_start = bucket_counts[:3].sum()
        tail = values[tail_start:]

        replacement_value = int(
            np.argmax(
                np.bincount(
                    tail.astype(np.int64)
                )
            )
        )

        candidate = np.flatnonzero(
            tail == replacement_value
        )[0]

        tail[candidate] = max_value
        values[tail_start:] = tail

    # Đổi một phần rất nhỏ demand=1 thành demand=0. Sau đó repair lại
    # tổng và tổng bình phương nên mean/std mục tiêu vẫn được giữ.
    zero_count = int(np.rint(n * ZERO_RATE_BY_MODE[mode]))
    one_positions = np.flatnonzero(values == 1)

    if zero_count > len(one_positions):
        raise RuntimeError(
            f"Not enough demand=1 rows to create zeros for {mode}"
        )

    zero_mark_positions = rng.choice(
        one_positions,
        size=zero_count,
        replace=False,
    )
    values[zero_mark_positions] = 0

    values = repair_integer_moments(
        values,
        mode,
    )

    rng.shuffle(values)

    assert len(values) == n
    return values


def assign_marks_with_sparse_zeros(
    marks,
    propensity,
    hex_index,
    time_index,
    rng,
):
    """Gán marks theo propensity và không để zero liên tiếp/cùng hex."""
    n_rows = len(marks)
    zero_count = int(np.count_nonzero(marks == 0))

    if zero_count == 0:
        order = np.argsort(propensity, kind="stable")
        demand = np.empty(n_rows, dtype=np.int16)
        demand[order] = np.sort(marks)
        return demand

    # Zero is sparse globally, mostly overnight in low-activity cells;
    # choose within the bottom propensity band so zero never lands at a peak.
    candidate_count = min(
        n_rows,
        max(
            zero_count * 20,
            int(np.ceil(n_rows * ZERO_CANDIDATE_SHARE)),
        ),
    )

    low_candidates = np.argpartition(
        propensity,
        candidate_count - 1,
    )[:candidate_count]
    slot = time_index[low_candidates] % SLOTS_PER_DAY
    overnight = (slot < 30) | (slot >= 138)
    night_candidates = low_candidates[overnight]
    other_candidates = low_candidates[~overnight]
    rng.shuffle(night_candidates)
    rng.shuffle(other_candidates)

    selected = []
    selected_keys = set()

    night_target = int(np.rint(0.85 * zero_count))
    for candidates, target in (
        (night_candidates, night_target),
        (other_candidates, zero_count),
        (night_candidates, zero_count),
    ):
        if len(selected) >= zero_count:
            break
        for row_index in candidates:
            hex_value = int(hex_index[row_index])
            time_value = int(time_index[row_index])
            key = hex_value * N_TIME + time_value
            if key in selected_keys or key - 1 in selected_keys or key + 1 in selected_keys:
                continue
            selected.append(int(row_index))
            selected_keys.add(key)
            if len(selected) >= target:
                break

    if len(selected) != zero_count:
        raise RuntimeError(
            f"Cannot place sparse zeros: selected={len(selected)}, "
            f"target={zero_count}, night_candidates={len(night_candidates)}, "
            f"other_candidates={len(other_candidates)}"
        )

    zero_positions = np.asarray(selected, dtype=np.int64)
    zero_mask = np.zeros(n_rows, dtype=bool)
    zero_mask[zero_positions] = True

    demand = np.empty(n_rows, dtype=np.int16)
    demand[zero_positions] = 0

    positive_positions = np.flatnonzero(~zero_mask)
    positive_order = positive_positions[
        np.argsort(
            propensity[positive_positions],
            kind="stable",
        )
    ]
    demand[positive_order] = np.sort(marks[marks > 0])

    return demand


def validate_generated_demand(
    demand,
    mode,
    hex_index,
    time_index,
):
    """Fail-fast nếu zero hoặc thống kê sinh ra lệch contract thực."""
    spec = CONTRACT[mode]
    expected_zero_count = int(
        np.rint(spec["n"] * ZERO_RATE_BY_MODE[mode])
    )

    actual_quantiles = np.quantile(
        demand,
        [0.50, 0.95, 0.99],
    )

    if len(demand) != spec["n"]:
        raise AssertionError(f"Invalid n for {mode}")
    if not np.isclose(demand.mean(), spec["mean"], atol=5e-7):
        raise AssertionError(f"Invalid mean for {mode}")
    if not np.isclose(demand.std(ddof=1), spec["std"], atol=5e-7):
        raise AssertionError(f"Invalid std for {mode}")
    if not np.array_equal(
        actual_quantiles,
        [spec["p50"], spec["p95"], spec["p99"]],
    ):
        raise AssertionError(f"Invalid quantiles for {mode}")
    if int(demand.max()) != spec["max"]:
        raise AssertionError(f"Invalid max for {mode}")
    if int(np.count_nonzero(demand == 0)) != expected_zero_count:
        raise AssertionError(f"Invalid zero count for {mode}")

    slot = time_index % SLOTS_PER_DAY
    zero = demand == 0
    overnight = (slot < 30) | (slot >= 138)
    if zero.any() and np.mean(overnight[zero]) < 0.80:
        raise AssertionError(f"Too few overnight zero rows for {mode}")
    if np.mean(demand[(slot >= 42) & (slot < 54)]) <= 2 * np.mean(
        demand[(slot >= 6) & (slot < 24)]
    ):
        raise AssertionError(f"Morning is not above overnight for {mode}")
    if np.mean(demand[(slot >= 102) & (slot < 120)]) <= 2 * np.mean(
        demand[(slot >= 6) & (slot < 24)]
    ):
        raise AssertionError(f"Evening is not above overnight for {mode}")
    valid_neighbors = (np.diff(hex_index) == 0) & (np.diff(time_index) == 1)
    if valid_neighbors.mean() < 0.995:
        raise AssertionError(f"Too many holes in observed cell histories for {mode}")

    actual_bucket_rows = np.asarray([
        np.count_nonzero((demand >= 0) & (demand <= 1)),
        np.count_nonzero((demand >= 2) & (demand <= 9)),
        np.count_nonzero((demand >= 10) & (demand <= 49)),
        np.count_nonzero(demand >= 50),
    ])

    if not np.array_equal(
        actual_bucket_rows,
        bucket_counts_for_mode(mode),
    ):
        raise AssertionError(f"Invalid bucket rows for {mode}")

    zero_positions = np.flatnonzero(demand == 0)
    zero_keys = np.sort(
        hex_index[zero_positions].astype(np.int64) * N_TIME
        + time_index[zero_positions].astype(np.int64)
    )

    if len(zero_keys) > 1:
        same_hex = (
            zero_keys[1:] // N_TIME
            == zero_keys[:-1] // N_TIME
        )
        consecutive = same_hex & (np.diff(zero_keys) == 1)

        if np.any(consecutive):
            raise AssertionError(
                f"Consecutive zero buckets detected for {mode}"
            )

# ------------------------------------------------------------
# CHỌN CHÍNH XÁC N VỊ TRÍ HEX × BUCKET
# Mỗi hex có tối đa một đoạn ngày không quan sát, bù nhau giữa các hex.
# ------------------------------------------------------------
def select_bucket_positions(n_rows, rng, mode):
    full_days, remainder = divmod(
        n_rows,
        SLOTS_PER_DAY,
    )
    min_days = 42
    if not N_HEX * min_days <= full_days <= N_HEX * N_DAYS:
        raise ValueError("n_rows cannot support the requested continuous history")

    # This controls observation coverage only; it does not change the
    # frequency of zero within the rows that are actually observed.
    mode_weight = (1.0 if mode == "Car" else 0.91)
    weights = np.exp(mode_weight * (SPATIAL_LOG_WEIGHT - SPATIAL_LOG_WEIGHT.mean()))
    days = np.full(N_HEX, min_days, dtype=np.int32)
    remaining = full_days - int(days.sum())
    while remaining:
        eligible = days < N_DAYS
        proportions = weights[eligible] / weights[eligible].sum()
        extra = np.minimum(N_DAYS - days[eligible], np.floor(remaining * proportions).astype(int))
        if not extra.any():
            indices = np.flatnonzero(eligible)
            priority = np.argsort(-proportions, kind="stable")
            extra[priority[:remaining]] = 1
        days[eligible] += extra
        remaining = full_days - int(days.sum())

    # Spread the missing interval evenly around the circular date axis. This
    # prevents a large artificial loss of hex coverage at the edges of the
    # four-month window; each hex has at most one absence interval.
    permuted_rank = rng.permutation(N_HEX)
    missing_starts = (
        (permuted_rank * N_DAYS // N_HEX)
        + int(rng.integers(0, N_DAYS))
    ) % N_DAYS
    hex_parts = []
    time_parts = []
    observed_days_by_hex = []
    for cell_idx, n_days in enumerate(days):
        missing_length = N_DAYS - int(n_days)
        observed_days = np.flatnonzero(
            ((np.arange(N_DAYS) - missing_starts[cell_idx]) % N_DAYS)
            >= missing_length
        )
        observed_days_by_hex.append(observed_days)
        size = len(observed_days) * SLOTS_PER_DAY
        hex_parts.append(np.full(size, cell_idx, dtype=np.int16))
        time_parts.append((
            np.repeat(observed_days, SLOTS_PER_DAY) * SLOTS_PER_DAY
            + np.tile(np.arange(SLOTS_PER_DAY), len(observed_days))
        ).astype(np.int32))

    if remainder:
        cell_idx = int(rng.choice(np.flatnonzero(days < N_DAYS)))
        obs = np.zeros(N_DAYS, dtype=bool)
        obs[observed_days_by_hex[cell_idx]] = True
        right_edge = np.flatnonzero(obs[:-1] & ~obs[1:])
        if len(right_edge):
            first = (int(right_edge[0]) + 1) * SLOTS_PER_DAY
        else:
            left_edge = np.flatnonzero(~obs[:-1] & obs[1:])
            first = (int(left_edge[0]) + 1) * SLOTS_PER_DAY - remainder
        hex_parts.append(np.full(remainder, cell_idx, dtype=np.int16))
        time_parts.append(np.arange(first, first + remainder, dtype=np.int32))

    hex_index = np.concatenate(hex_parts)
    time_index = np.concatenate(time_parts)
    order = np.lexsort((time_index, hex_index))
    return hex_index[order], time_index[order]


# ------------------------------------------------------------
# QUY LUẬT THỜI GIAN
# ------------------------------------------------------------
def time_profile(mode, time_index, hex_index):
    day_index = time_index // SLOTS_PER_DAY
    slot = time_index % SLOTS_PER_DAY

    hour = slot / 6.0

    day_of_week = (
        LOCAL_START.dayofweek + day_index
    ) % 7

    working_day = day_of_week < 5

    morning = np.exp(
        -0.5 * ((hour - 8.0) / 1.4) ** 2
    )
    lunch = np.exp(
        -0.5 * ((hour - 12.0) / 2.1) ** 2
    )
    evening = np.exp(
        -0.5 * ((hour - 18.0) / 1.6) ** 2
    )
    night = np.exp(
        -0.5 * ((hour - 21.5) / 2.0) ** 2
    )

    if mode == "Car":
        weekday = (
            0.42
            + 1.20 * morning
            + 0.35 * lunch
            + 1.55 * evening
            + 0.18 * night
        )

        weekend = (
            0.52
            + 0.45 * morning
            + 0.85 * lunch
            + 1.00 * evening
            + 0.30 * night
        )
    else:
        weekday = (
            0.45
            + 1.35 * morning
            + 0.48 * lunch
            + 1.70 * evening
            + 0.23 * night
        )

        weekend = (
            0.55
            + 0.55 * morning
            + 0.95 * lunch
            + 1.15 * evening
            + 0.38 * night
        )

    baseline = np.where(
        working_day,
        weekday,
        weekend,
    )
    # Pickup origins can differ across spatial zones; these are heuristics.
    zone = ZONE[hex_index]
    early = np.exp(-0.5 * ((hour - 7.7) / 1.35) ** 2)
    late = np.exp(-0.5 * ((hour - 18.2) / 1.5) ** 2)
    after_dark = np.exp(-0.5 * ((hour - 22.5) / 1.9) ** 2)
    # Airport cells retain activity overnight; outer cells are very quiet.
    baseline += (zone == "airport") * (
        0.22 + 0.20 * early + 0.18 * after_dark
    )
    baseline += (zone == "central") * (
        0.11 + 0.15 * late + 0.27 * after_dark * (~working_day)
    )
    baseline += ((zone == "south") | (zone == "west")) * (
        0.16 * early * working_day + 0.08 * late
    )
    baseline += (zone == "east") * (0.13 * early * working_day)
    overnight = (hour < 5.0)
    night_factor = np.where(
        zone == "outer", 0.27,
        np.where(zone == "airport", 0.78,
                 np.where(zone == "central", 0.54, 0.38))
    )
    # Demand 00:00–05:00 is often 0/1/2 in lower-activity cells.
    return baseline * np.where(overnight, night_factor, 1.0)


def generate_fake_data(
    output_dir: str | Path | None = None,
    overwrite: bool = True,
    verbose: bool = False,
) -> tuple[Path, pd.DataFrame]:
    """Write the notebook's continuous 10-minute Car/Motorcycle fake history."""
    OUTPUT_DIR = Path(output_dir) if output_dir is not None else PROJECT_ROOT / "data" / "raw"
    OUTPUT_DIR = OUTPUT_DIR.expanduser().resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEMAND_PATH = OUTPUT_DIR / "xanhsm_hcm_smooth_10min_bucket_v9.parquet"
    OUTPUT_SCHEMA = pa.schema([
        pa.field(
            "period_datetime_utc",
            pa.timestamp("us", tz="UTC"),
        ),
        pa.field("hex_id_7", pa.string()),
        pa.field("travel_mode", pa.string()),
        pa.field("total_demand", pa.int16()),
        pa.field("is_holiday", pa.int8()),
    ])

    if DEMAND_PATH.exists():
        if not overwrite:
            raise FileExistsError(DEMAND_PATH)
        DEMAND_PATH.unlink()

    writer = pq.ParquetWriter(
        DEMAND_PATH,
        OUTPUT_SCHEMA,
        compression="zstd",
        use_dictionary=[
            "hex_id_7",
            "travel_mode",
        ],
    )

    shared_rng = np.random.default_rng(SEED)

    hex_to_index = {hex_id: i for i, hex_id in enumerate(COMMON_HEX)}
    hex_coverage = []

    # Trạng thái thành phố thay đổi chậm theo ngày
    city_state = shared_rng.normal(
        0,
        1,
        size=N_DAYS,
    )

    city_state = gaussian_filter1d(
        city_state,
        sigma=2.0,
    )

    city_state = (
        city_state - city_state.mean()
    ) / (city_state.std() + 1e-8)

    START_UTC_US = int(
        LOCAL_START.tz_convert("UTC").value
        // 1_000
    )

    TEN_MINUTES_US = (
        10 * 60 * 1_000_000
    )

    hex_array = np.asarray(
        COMMON_HEX,
        dtype=object,
    )

    generation_summary = []

    try:
        for mode_index, mode in enumerate(
            VALID_MODES
        ):
            rng = np.random.default_rng(
                SEED + 10_000 * (mode_index + 1)
            )

            n_rows = CONTRACT[mode]["n"]

            hex_index, time_index = (
                select_bucket_positions(
                    n_rows,
                    rng,
                    mode,
                )
            )

            # Demand marks giữ đúng thống kê tổng
            marks = create_calibrated_marks(
                mode,
                rng,
            )

            # Trạng thái demand mịn theo từng hex
            smooth_state = rng.normal(
                0,
                1,
                size=(N_HEX, N_TIME),
            ).astype(np.float32)

            smooth_state = gaussian_filter1d(
                smooth_state,
                sigma=8.0,  # khoảng 80 phút
                axis=1,
                mode="nearest",
            )

            smooth_state = (
                smooth_state
                - smooth_state.mean(
                    axis=1,
                    keepdims=True,
                )
            ) / (
                smooth_state.std(
                    axis=1,
                    keepdims=True,
                )
                + 1e-6
            )

            day_index = (
                time_index // SLOTS_PER_DAY
            )

            is_holiday = IS_HOLIDAY_BY_DAY[day_index].astype(np.int8)

            # --------------------------------------------------------
            # HOLIDAY EFFECT — tăng NHẸ, không tạo spike quá giả
            # --------------------------------------------------------
            # Chỉ thay đổi propensity nên bộ marks / phân phối tổng vẫn giữ contract.
            # Holiday effect tập trung từ trưa đến tối vì nhu cầu đi lại thường rõ hơn
            # ở các khung giờ này; sáng sớm chỉ được boost rất nhẹ.
            slot = time_index % SLOTS_PER_DAY
            hour = slot / 6.0

            holiday_time_profile = (
                0.30
                + 0.25
                * np.exp(
                    -0.5 * ((hour - 12.0) / 2.7) ** 2
                )
                + 0.45
                * np.exp(
                    -0.5 * ((hour - 18.5) / 3.0) ** 2
                )
            )

            # Có thể tăng/giảm 2 số này để holiday nổi bật hơn/ít hơn.
            # Giá trị hiện tại chỉ làm holiday nhô lên vừa phải.
            HOLIDAY_STRENGTH = {
                "Car": 0.06,
                "Motorcycle": 0.06,
            }[mode]

            holiday_propensity_boost = (
                is_holiday
                * HOLIDAY_STRENGTH
                * holiday_time_profile
            )

            mode_spatial_effect = (
                SPATIAL_LOG_WEIGHT
                + rng.normal(0, 0.09, size=N_HEX)
            )

            # Localized, time-limited events introduce rare peaks without
            # altering the exact overall demand histogram / maximum.
            event_state = np.zeros((N_HEX, N_TIME), dtype=np.float32)
            for _ in range(75):
                focus = int(rng.integers(0, N_HEX))
                start = int(rng.integers(0, N_TIME - 14))
                neighbors = [
                    hex_to_index[h]
                    for h in h3.grid_disk(COMMON_HEX[focus], 1)
                    if h in hex_to_index
                ]
                length = int(rng.integers(5, 15))
                shape = np.sin(np.linspace(0, np.pi, length, dtype=np.float32))
                event_state[np.ix_(neighbors, np.arange(start, start + length))] += (
                    rng.uniform(0.6, 1.35) * shape[None, :]
                )

            profile = time_profile(
                mode,
                time_index,
                hex_index,
            )

            # Noise nhỏ (0.08 như bản đầu) vì dữ liệu bucket thực đã mịn
            propensity = (
                1.15
                * mode_spatial_effect[hex_index]
                + 1.15
                * np.log(profile + 1e-6)
                + 0.09
                * city_state[day_index]
                + 0.44
                * smooth_state[
                    hex_index,
                    time_index,
                ]
                + event_state[hex_index, time_index]
                + holiday_propensity_boost
                + 0.06
                * time_index
                / max(N_TIME - 1, 1)
                + rng.normal(0, 0.08, size=n_rows)
            )

            # Demand lớn đi vào propensity lớn; zero chỉ nằm trong vùng
            # propensity thấp và không liên tiếp trên cùng một hex.
            demand = assign_marks_with_sparse_zeros(
                marks,
                propensity,
                hex_index,
                time_index,
                rng,
            )

            validate_generated_demand(
                demand,
                mode,
                hex_index,
                time_index,
            )

            # Separate spatial lookup to interpret the H3 identifiers without
            # leaking synthetic zone labels into the existing ML training schema.
            coverage = HEX_METADATA.copy()
            coverage["travel_mode"] = mode
            coverage["observed_rows"] = np.bincount(hex_index, minlength=N_HEX)
            coverage["first_observed_local"] = (
                LOCAL_START + pd.to_timedelta(
                    10 * np.minimum.reduceat(
                        time_index,
                        np.r_[0, np.cumsum(coverage["observed_rows"].to_numpy())[:-1]],
                    ),
                    unit="min",
                )
            )
            coverage["zero_rows"] = np.bincount(
                hex_index[demand == 0], minlength=N_HEX,
            )
            night_mask = ((time_index % SLOTS_PER_DAY) < 30) | (
                (time_index % SLOTS_PER_DAY) >= 138
            )
            coverage["night_rows"] = np.bincount(
                hex_index[night_mask], minlength=N_HEX,
            )
            coverage["night_demand_sum"] = np.bincount(
                hex_index[night_mask], weights=demand[night_mask], minlength=N_HEX,
            ).astype(np.int64)
            for demand_value in (0, 1, 2):
                coverage[f"night_{demand_value}_rows"] = np.bincount(
                    hex_index[night_mask & (demand == demand_value)], minlength=N_HEX,
                )
            hex_coverage.append(coverage)

            timestamp_us = (
                START_UTC_US
                + time_index.astype(np.int64)
                * TEN_MINUTES_US
            )

            generation_summary.append({
                "travel_mode": mode,
                "n": len(demand),
                "mean": demand.mean(),
                "std": demand.std(ddof=1),
                "CV": (
                    demand.std(ddof=1)
                    / demand.mean()
                ),
                "p50": np.quantile(
                    demand,
                    0.50,
                ),
                "p95": np.quantile(
                    demand,
                    0.95,
                ),
                "p99": np.quantile(
                    demand,
                    0.99,
                ),
                "max": demand.max(),
                "zero_rows": int(
                    np.count_nonzero(demand == 0)
                ),
                "zero_pct": (
                    100.0
                    * np.count_nonzero(demand == 0)
                    / len(demand)
                ),
                "max_consecutive_zero_buckets": (
                    1 if np.any(demand == 0) else 0
                ),
                "number_of_hex": len(
                    np.unique(hex_index)
                ),
                "overnight_zero_pct": round(100 * np.mean(
                    ((time_index[demand == 0] % SLOTS_PER_DAY) < 30)
                    | ((time_index[demand == 0] % SLOTS_PER_DAY) >= 138)
                ), 2),
                "morning_mean": round(float(demand[
                    ((time_index % SLOTS_PER_DAY) >= 42)
                    & ((time_index % SLOTS_PER_DAY) < 54)
                ].mean()), 2),
                "late_night_mean": round(float(demand[
                    ((time_index % SLOTS_PER_DAY) >= 6)
                    & ((time_index % SLOTS_PER_DAY) < 24)
                ].mean()), 2),
                "holiday_rows": int(is_holiday.sum()),
                "holiday_mean": (
                    float(demand[is_holiday == 1].mean())
                    if np.any(is_holiday == 1)
                    else np.nan
                ),
                "non_holiday_mean": (
                    float(demand[is_holiday == 0].mean())
                    if np.any(is_holiday == 0)
                    else np.nan
                ),
            })

            chunk_size = 250_000

            for start in range(
                0,
                n_rows,
                chunk_size,
            ):
                stop = min(
                    start + chunk_size,
                    n_rows,
                )

                table = pa.Table.from_arrays(
                    [
                        pa.array(
                            timestamp_us[start:stop],
                            type=pa.timestamp(
                                "us",
                                tz="UTC",
                            ),
                        ),
                        pa.array(
                            hex_array[
                                hex_index[start:stop]
                            ]
                        ),
                        pa.array(
                            np.repeat(
                                mode,
                                stop - start,
                            )
                        ),
                        pa.array(
                            demand[start:stop],
                            type=pa.int16(),
                        ),
                        pa.array(
                            is_holiday[start:stop],
                            type=pa.int8(),
                        ),
                    ],
                    schema=OUTPUT_SCHEMA,
                )

                writer.write_table(table)

            del (
                marks,
                hex_index,
                time_index,
                smooth_state,
                event_state,
                propensity,
                demand,
                is_holiday,
                holiday_propensity_boost,
                timestamp_us,
            )

            gc.collect()

    finally:
        writer.close()

    GENERATION_SUMMARY = pd.DataFrame(generation_summary)
    HEX_METADATA_PATH = OUTPUT_DIR / "hcm_hex_synthetic_profiles_v11.csv"
    pd.concat(hex_coverage, ignore_index=True).to_csv(HEX_METADATA_PATH, index=False)
    if verbose:
        print(GENERATION_SUMMARY.to_string(index=False))
        print("H3 profile:", HEX_METADATA_PATH)
        print("Parquet:", DEMAND_PATH)
    return DEMAND_PATH, GENERATION_SUMMARY


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Generate the notebook's synthetic demand")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-overwrite", action="store_true")
    args = parser.parse_args()
    path, summary = generate_fake_data(args.output_dir, not args.no_overwrite, verbose=False)
    print(summary.to_string(index=False))
    print(path)


if __name__ == "__main__":
    main()
