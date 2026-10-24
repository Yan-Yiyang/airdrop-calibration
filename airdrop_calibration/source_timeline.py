"""标定使用与在线一致的源时标配对和四元数接收参照，不导入运行包。"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from .algorithms import _slerp, quaternion_to_matrix_local
from .models import Telemetry


@dataclass(frozen=True)
class _Sample:
    source_us: int
    received_at: float
    values: tuple[float, ...]


def _samples(
    rows: Sequence[Telemetry],
    fields: tuple[str, ...],
    stamp: str,
    received: str,
    *,
    require_timestamps: bool = True,
) -> list[_Sample]:
    """去掉合并快照重复持有的字段；标定只接受连续、同一钟段的数据。"""
    result: list[_Sample] = []
    for row in rows:
        values = tuple(getattr(row, name) for name in fields)
        if any(value is None for value in values):
            continue
        source, wall = getattr(row, stamp), getattr(row, received)
        if source is None or wall is None:
            if not require_timestamps:
                continue
            raise ValueError(f"标定记录缺少 {stamp}/{received}，必须重新录制带源时标的数据")
        if source <= 0:
            continue
        if result and source == result[-1].source_us:
            continue
        if result and (source < result[-1].source_us or wall <= result[-1].received_at):
            raise ValueError("标定记录跨越时钟回退或重启，请选择连续的同一钟段")
        result.append(_Sample(source, wall, values))
    return result


def _pair(
    samples: list[_Sample],
    value: float,
    *,
    received: bool = False,
    max_source_gap_us: int | None = 300_000,
):
    """按源时标或接收时刻找相邻样本，拒绝外推与超过在线门限的间隙。"""
    key = (lambda item: item.received_at) if received else (lambda item: item.source_us)
    if len(samples) < 2 or value < key(samples[0]) or value > key(samples[-1]):
        return None
    index = min(max(bisect_right(samples, value, key=key), 1), len(samples) - 1)
    left, right = samples[index - 1], samples[index]
    gap = right.source_us - left.source_us
    if gap <= 0 or (max_source_gap_us is not None and gap > max_source_gap_us):
        return None
    if received and not 0 < right.received_at - left.received_at <= 0.5:
        return None
    return left, right, (value - key(left)) / (key(right) - key(left))


class SourceTimeline:
    """位置按源时刻对齐姿态；墙钟参照始终为四元数的原始接收时刻。"""

    def __init__(self, rows: Sequence[Telemetry], *, include_position: bool = True) -> None:
        self._attitude = _samples(
            rows,
            tuple("quaternion_" + axis for axis in "wxyz"),
            "quaternion_timestamp_us",
            "quaternion_received_at",
        )
        self._position = (
            _samples(
                rows,
                ("north_m", "east_m", "down_m"),
                "position_timestamp_us",
                "position_velocity_received_at",
                require_timestamps=False,
            )
            if include_position
            else []
        )
        self._include_position = include_position
        if len(self._attitude) < 2:
            raise ValueError("标定需要至少两个带独立源时标和接收时刻的四元数样本")
        for left, right in pairwise(self._attitude):
            if not 0 < right.source_us - left.source_us <= 300_000 or not (
                0 < right.received_at - left.received_at <= 0.5
            ):
                raise ValueError("四元数记录存在超出在线插值门限的断流，请使用连续片段标定")
        self.attitudes = [
            Telemetry(
                timestamp=sample.received_at,
                quaternion_w=sample.values[0],
                quaternion_x=sample.values[1],
                quaternion_y=sample.values[2],
                quaternion_z=sample.values[3],
            )
            for sample in self._attitude
        ]

    @property
    def quality_meta(self) -> dict[str, dict[str, float | int | bool | None]]:
        """返回位置和四元数各自的去重数量、频率与源采样间隙。"""
        result: dict[str, dict[str, float | int | None]] = {}
        for name, samples in (("position", self._position), ("quaternion", self._attitude)):
            source_gaps = sorted(
                right.source_us - left.source_us for left, right in pairwise(samples)
            )
            received_gaps = sorted(
                right.received_at - left.received_at for left, right in pairwise(samples)
            )
            count = len(samples)
            span = samples[-1].source_us - samples[0].source_us if count >= 2 else 0
            received_span = samples[-1].received_at - samples[0].received_at if count >= 2 else 0.0
            source_p95 = (
                source_gaps[max(math.ceil(0.95 * len(source_gaps)) - 1, 0)] if source_gaps else None
            )
            received_p95 = (
                received_gaps[max(math.ceil(0.95 * len(received_gaps)) - 1, 0)]
                if received_gaps
                else None
            )
            result[name] = {
                "used": self._include_position if name == "position" else True,
                "native_samples": count,
                "source_hz": ((count - 1) * 1_000_000 / span) if span > 0 else None,
                "source_gap_median_ms": float(np.median(source_gaps) / 1000)
                if source_gaps
                else None,
                "source_gap_p95_ms": (source_p95 / 1000) if source_p95 is not None else None,
                "source_gap_max_ms": (source_gaps[-1] / 1000) if source_gaps else None,
                "received_hz": ((count - 1) / received_span) if received_span > 0 else None,
                "received_gap_median_ms": float(np.median(received_gaps) * 1000)
                if received_gaps
                else None,
                "received_gap_p95_ms": (received_p95 * 1000) if received_p95 is not None else None,
                "received_gap_max_ms": (received_gaps[-1] * 1000) if received_gaps else None,
            }
        return result

    def pose_at(self, timestamp: float) -> np.ndarray | None:
        """与在线查询同样先映射到源时刻，再分别内插位置与姿态。"""
        anchors = _pair(self._attitude, timestamp, received=True)
        if anchors is None:
            return None
        left, right, weight = anchors
        source = left.source_us + weight * (right.source_us - left.source_us)
        position = _pair(self._position, source, max_source_gap_us=None)
        if position is None:
            return None
        a, b, ratio = position
        quaternion = _slerp(left.values, right.values, weight)
        pose = np.eye(4)
        pose[:3, :3] = quaternion_to_matrix_local(tuple(quaternion))
        pose[:3, 3] = (1 - ratio) * np.asarray(a.values) + ratio * np.asarray(b.values)
        return pose

    def rotation_at(self, timestamp: float) -> np.ndarray | None:
        """只按四元数接收参照映射源时刻并插值旋转，不依赖位置遥测。"""
        anchors = _pair(self._attitude, timestamp, received=True)
        if anchors is None:
            return None
        left, right, weight = anchors
        quaternion = _slerp(left.values, right.values, weight)
        return quaternion_to_matrix_local(tuple(quaternion))
