"""标定时间参照：合并快照更新不能改变所测的对齐量。"""

import math
from dataclasses import replace

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
