"""标定时间参照：合并快照更新不能改变所测的对齐量。"""

import math
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

from airdrop_calibration.algorithms import measure_frame_telemetry_lag, rotation_z
from airdrop_calibration.models import Telemetry
from airdrop_calibration.source_timeline import SourceTimeline


@pytest.mark.parametrize("merged_extra", [0.0, 0.12, 0.27])
def test_lag_uses_quaternion_reference_and_not_merged_update(merged_extra: float) -> None:
    physical = np.arange(0, 6, 0.02) + 100
    angles = 20 * np.sin((physical - 100) * 2.3) + 7 * np.sin((physical - 100) * 5.1)
    rows = [
        Telemetry(
            timestamp=float(t + 0.07 + merged_extra),
            quaternion_received_at=float(t + 0.07),
            quaternion_timestamp_us=round((t - 99) * 1e6),
            quaternion_w=math.cos(math.radians(angle) / 2),
            quaternion_x=0.0,
            quaternion_y=0.0,
            quaternion_z=math.sin(math.radians(angle) / 2),
        )
        for t, angle in zip(physical, angles, strict=True)
    ]
    # 同一源样本被其它字段重复持有，不得变成新的姿态观测。
    held = [replace(row, timestamp=row.timestamp + 0.001) for row in rows]
    timeline = SourceTimeline([item for pair in zip(rows, held, strict=True) for item in pair])
    measurement = measure_frame_telemetry_lag(
        list(physical + 0.35),
        list(physical + 0.15),
        [rotation_z(angle) for angle in angles],
        timeline.attitudes,
        baseline_s=0.2,
        search_s=0.5,
        step_s=0.005,
    )
    assert len(timeline.attitudes) == len(rows)
    assert measurement.absolute.lag_s == pytest.approx(0.35 - 0.07, abs=0.015)
    assert measurement.residual is not None
    assert measurement.residual.lag_s + 0.2 == pytest.approx(measurement.absolute.lag_s, abs=0.015)


def test_missing_native_timestamps_is_an_error() -> None:
    with pytest.raises(ValueError, match="重新录制"):
        SourceTimeline(
            [Telemetry(timestamp=1, quaternion_w=1, quaternion_x=0, quaternion_y=0, quaternion_z=0)]
        )


def test_discontinuous_quaternion_clock_is_an_error() -> None:
    rows = [
        Telemetry(
            timestamp=t,
            quaternion_received_at=t,
            quaternion_timestamp_us=stamp,
            quaternion_w=1,
            quaternion_x=0,
            quaternion_y=0,
            quaternion_z=0,
        )
        for t, stamp in [(100, 1_000_000), (101, 1_100_000)]
    ]
    with pytest.raises(ValueError, match="断流"):
        SourceTimeline(rows)


def test_source_clock_uses_positive_native_timestamps() -> None:
    rows = [
        Telemetry(
            timestamp=received,
            quaternion_received_at=received,
            quaternion_timestamp_us=1_000_000 + index * 100_000,
            quaternion_w=1.0,
            quaternion_x=0.0,
            quaternion_y=0.0,
            quaternion_z=0.0,
            position_velocity_received_at=received,
            position_timestamp_us=1_000_000 + index * 100_000,
            north_m=float(index),
            east_m=0.0,
            down_m=0.0,
        )
        for index, received in enumerate((10.0, 10.1))
    ]
    timeline = SourceTimeline(rows)
    assert timeline.quality_meta["quaternion"]["native_samples"] == 2
    assert timeline.quality_meta["position"]["native_samples"] == 2
    assert timeline.pose_at(10.05) is not None


def test_rotation_query_does_not_require_position_samples() -> None:
    rows = [
        Telemetry(
            timestamp=received,
            quaternion_received_at=received,
            quaternion_timestamp_us=1_000_000 + index * 100_000,
            quaternion_w=math.cos(math.radians(angle) / 2),
            quaternion_x=0.0,
            quaternion_y=0.0,
            quaternion_z=math.sin(math.radians(angle) / 2),
        )
        for index, (received, angle) in enumerate(((20.0, 0.0), (20.1, 20.0)))
    ]
    timeline = SourceTimeline(rows)
    assert timeline.pose_at(20.05) is None
    rotation = timeline.rotation_at(20.05)
    assert rotation is not None
    assert rotation == pytest.approx(rotation_z(10.0))


def test_rotation_query_ignores_invalid_position_clock() -> None:
    rows = [
        Telemetry(
            timestamp=10.0 + index * 0.1,
            quaternion_received_at=10.0 + index * 0.1,
            quaternion_timestamp_us=1_000_000 + index * 100_000,
            quaternion_w=1.0,
            quaternion_x=0.0,
            quaternion_y=0.0,
            quaternion_z=0.0,
            position_velocity_received_at=float(index),
            position_timestamp_us=None if index == 0 else 1_000_000,
            north_m=float(index),
            east_m=0.0,
            down_m=0.0,
        )
        for index in range(2)
    ]
    timeline = SourceTimeline(rows, include_position=False)
    assert timeline.rotation_at(10.05) is not None
    assert timeline.quality_meta["position"]["used"] is False


def test_bad_position_groups_do_not_drop_quaternion_timeline() -> None:
    rows = [
        Telemetry(
            timestamp=20.0 + index * 0.1,
            quaternion_received_at=20.0 + index * 0.1,
            quaternion_timestamp_us=1_000_000 + index * 100_000,
            quaternion_w=1.0,
            quaternion_x=0.0,
            quaternion_y=0.0,
            quaternion_z=0.0,
            north_m=cast(Any, "1.0") if index == 0 else float("nan"),
            east_m=2.0,
            down_m=3.0,
            position_timestamp_us=cast(Any, "bad") if index == 0 else 1_100_000,
            position_velocity_received_at=cast(Any, "bad") if index == 0 else 20.1,
        )
        for index in range(2)
    ]
    assert rows[0].quaternion_w == 1.0
    assert rows[0].north_m == 1.0
    assert rows[1].quaternion_w == 1.0
    assert rows[1].north_m is None
    timeline = SourceTimeline(rows, include_position=False)
    assert len(timeline.attitudes) == 2
    assert timeline.rotation_at(20.05) is not None


def test_numeric_string_position_values_are_normalized() -> None:
    row = Telemetry(
        timestamp=1.0,
        north_m=cast(Any, "1.0"),
        east_m=cast(Any, "2.0"),
        down_m=cast(Any, "3.0"),
        position_timestamp_us=10,
        position_velocity_received_at=cast(Any, "1.0"),
    )
    assert row.north_m == 1.0
    assert isinstance(row.north_m, float)
    assert row.position_velocity_received_at == 1.0


def test_position_interpolation_allows_one_hz_samples() -> None:
    rows = [
        Telemetry(
            timestamp=10.0 + index * 0.1,
            quaternion_received_at=10.0 + index * 0.1,
            quaternion_timestamp_us=1_000_000 + index * 100_000,
            quaternion_w=1.0,
            quaternion_x=0.0,
            quaternion_y=0.0,
            quaternion_z=0.0,
            position_timestamp_us=1_000_000 if index < 10 else 2_000_000,
            position_velocity_received_at=10.0 if index < 10 else 11.0,
            north_m=0.0 if index < 10 else 1.0,
            east_m=0.0,
            down_m=0.0,
        )
        for index in range(11)
    ]
    timeline = SourceTimeline(rows)
    pose = timeline.pose_at(10.5)
    assert pose is not None
    assert pose[:3, 3] == pytest.approx([0.5, 0.0, 0.0])


def test_quality_meta_counts_native_position_and_quaternion_samples_separately() -> None:
    rows = []
    for index in range(101):
        received = 10.0 + index * 0.01
        row = Telemetry(
            timestamp=received,
            position_timestamp_us=1_000_000 + (index // 2) * 20_000,
            position_velocity_received_at=10.0 + (index // 2) * 0.02,
            north_m=float(index // 2),
            east_m=0.0,
            down_m=0.0,
        )
        # 四元数每秒更新 100 次，位置每秒更新 50 次；合并快照重复持有位置字段。
        q_index = index
        row = replace(
            row,
            quaternion_timestamp_us=1_000_000 + q_index * 10_000,
            quaternion_received_at=received,
            quaternion_w=1.0,
            quaternion_x=0.0,
            quaternion_y=0.0,
            quaternion_z=0.0,
        )
        rows.append(row)
    timeline = SourceTimeline(rows)
    quality = timeline.quality_meta
    assert quality["quaternion"]["native_samples"] == 101
    assert quality["quaternion"]["source_hz"] == pytest.approx(100.0)
    assert quality["position"]["native_samples"] == 51
    assert quality["position"]["source_hz"] == pytest.approx(50.0)
    assert quality["quaternion"]["received_hz"] == pytest.approx(100.0)
    assert quality["position"]["received_hz"] == pytest.approx(50.0)
    assert quality["quaternion"]["source_gap_max_ms"] == pytest.approx(10.0)
    assert quality["quaternion"]["received_gap_max_ms"] == pytest.approx(10.0)
