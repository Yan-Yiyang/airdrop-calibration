"""标定工具自己的数据传输对象与结果类型。"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

TIME_BASE_RECEIVED = "received"
TIME_BASE_CAPTURE = "capture"


class TelemetryLike(Protocol):
    """算法所需的遥测字段；运行包快照可直接按结构传入。"""

    @property
    def timestamp(self) -> float: ...

    @property
    def north_m(self) -> float | None: ...

    @property
    def east_m(self) -> float | None: ...

    @property
    def down_m(self) -> float | None: ...

    @property
    def quaternion_w(self) -> float | None: ...

    @property
    def quaternion_x(self) -> float | None: ...

    @property
    def quaternion_y(self) -> float | None: ...

    @property
    def quaternion_z(self) -> float | None: ...


@dataclass(frozen=True, slots=True)
class Telemetry:
    """离线标定使用的遥测字段子集，单位为秒、米与无量纲四元数。"""

    timestamp: float = field(default_factory=time.time)
    north_m: float | None = None
    east_m: float | None = None
    down_m: float | None = None
    quaternion_w: float | None = None
    quaternion_x: float | None = None
    quaternion_y: float | None = None
    quaternion_z: float | None = None
    quaternion_timestamp_us: int | None = None
    quaternion_received_at: float | None = None
    position_timestamp_us: int | None = None
    position_velocity_received_at: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.timestamp, bool):
            raise ValueError("遥测 timestamp 必须是有限数值")
        timestamp = float(self.timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("遥测 timestamp 必须是有限数值")
        object.__setattr__(self, "timestamp", timestamp)
        for name in (
            "north_m",
            "east_m",
            "down_m",
            "quaternion_w",
            "quaternion_x",
            "quaternion_y",
            "quaternion_z",
            "quaternion_received_at",
            "position_velocity_received_at",
        ):
            value = getattr(self, name)
            if value is not None:
                if isinstance(value, bool):
                    raise ValueError(f"遥测 {name} 必须是有限数值")
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError(f"遥测 {name} 必须是有限数值")
                object.__setattr__(self, name, number)
        for name in ("quaternion_timestamp_us", "position_timestamp_us"):
            value = getattr(self, name)
            if value is not None:
                if isinstance(value, bool) or not math.isfinite(value) or int(value) != value:
                    raise ValueError(f"遥测 {name} 必须是整数源时标")
                object.__setattr__(self, name, int(value))

    def is_valid(self) -> bool:
        """存在位置或姿态观测时返回真。"""
        return any(
            value is not None
            for value in (
                self.north_m,
                self.quaternion_w,
            )
        )


@dataclass
class IntrinsicsResult:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: tuple[int, int]
    rms: float
    views: int
    per_view_errors: list[float] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.views >= 3 and math.isfinite(self.rms)


@dataclass
class LagResult:
    lag_s: float
    peak: float
    samples: int
    curve: list[tuple[float, float]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.samples > 0 and math.isfinite(self.lag_s)


@dataclass(frozen=True, slots=True)
class FrameStamp:
    """索引记录中的收到时刻、录制假设时刻和当时配置的延时。"""

    received_timestamp: float
    capture_timestamp: float
    lag_s: float

    def __post_init__(self) -> None:
        for name in ("received_timestamp", "capture_timestamp", "lag_s"):
            raw_value = getattr(self, name)
            if isinstance(raw_value, bool):
                raise ValueError(f"帧索引 {name} 必须是有限数值")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(f"帧索引 {name} 必须是有限数值")
            object.__setattr__(self, name, value)


@dataclass
class LagMeasurement:
    absolute: LagResult
    residual: LagResult | None
    baseline_s: float
    cross_check_ok: bool
    time_base: str = TIME_BASE_RECEIVED


@dataclass
class ExtrinsicsResult:
    r_bc: np.ndarray
    t_bc: np.ndarray | None
    pairs: int
    residual_rotation_deg: float
    axis_spread_deg: float = 0.0
    t_bc_reliable: bool = False
    residual_translation_m: float = float("nan")
    body_span_m: float = 0.0
    camera_span_m: float = 0.0
    body_span_ok: bool = False
    methods: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.pairs >= 3 and math.isfinite(self.residual_rotation_deg)
