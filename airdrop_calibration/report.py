"""生成兼容 CameraModel 的标定 JSON 载荷与来源元数据。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from .models import ExtrinsicsResult, IntrinsicsResult, LagMeasurement, LagResult
from .settings import LAG_CROSS_CHECK_TOL_S, MEASURED_T_BC, TIME_BASE_RECEIVED

CalibrationSource = Literal["measured_calibration", "simulated_calibration"]
SCHEMA_VERSION = 1


def build_calibration_report(
    *,
    intrinsics: IntrinsicsResult,
    lag: LagResult,
    measurement: LagMeasurement,
    extrinsics: ExtrinsicsResult | None,
    flight_dir: Path,
    pattern_size: tuple[int, int],
    square_size: float,
    lag_search_s: float,
    lag_step_s: float,
    measured_t_bc: Sequence[float] = MEASURED_T_BC,
    source: CalibrationSource = "measured_calibration",
    telemetry_quality: Mapping[str, Any] | None = None,
    translation_diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    """构建标定文件；未求得的旋转外参明确记为 null。"""
    if source not in ("measured_calibration", "simulated_calibration"):
        raise ValueError(f"未知标定来源: {source!r}")
    return {
        "camera_matrix": intrinsics.camera_matrix.tolist(),
        "dist_coeffs": intrinsics.dist_coeffs.tolist(),
        "image_size": list(intrinsics.image_size),
        "R_bc": extrinsics.r_bc.tolist() if extrinsics is not None else None,
        "t_bc": [float(value) for value in measured_t_bc],
        "telemetry_lag": float(lag.lag_s) if lag.ok else None,
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "source": source,
            "units": {
                "camera_matrix": "pixel",
                "dist_coeffs": "dimensionless",
                "image_size": "pixel (width, height)",
                "R_bc": "dimensionless rotation matrix",
                "t_bc": "meter",
                "telemetry_lag": "second",
            },
            "coordinates": {
                "R_bc": "camera optical frame to body FRD frame",
                "t_bc": "camera optical center position expressed in body FRD frame",
                "telemetry_position": "north-east-down in local NED navigation frame",
                "telemetry_quaternion": "body FRD frame to local NED navigation frame",
                "time_base": TIME_BASE_RECEIVED,
            },
            "intrinsics": {
                "rms_px": intrinsics.rms,
                "views": intrinsics.views,
                "per_view_errors": intrinsics.per_view_errors,
                "pattern_size": list(pattern_size),
                "square_size_m": square_size,
            },
            "time_offset": {
                "lag_s": lag.lag_s if lag.ok else None,
                "peak_correlation": lag.peak if lag.ok else None,
                "samples": lag.samples,
                "search_s": lag_search_s,
                "step_s": lag_step_s,
                "curve": lag.curve if lag.ok else [],
                "time_base": measurement.time_base,
                "telemetry_reference": "quaternion_received_at",
                "pose_alignment": "source_timestamp_us",
                "lag_baseline_s": measurement.baseline_s,
                "lag_residual_on_capture_axis_s": (
                    float(measurement.residual.lag_s)
                    if measurement.residual is not None and measurement.residual.ok
                    else None
                ),
                "cross_check_ok": bool(measurement.cross_check_ok),
                "cross_check_tolerance_s": LAG_CROSS_CHECK_TOL_S,
                "note": (
                    "Δt 使用收到帧时刻 received_timestamp，是绝对对齐量；"
                    "capture 轴残差加录制基准应与该绝对量相符"
                ),
            },
            "extrinsics": (
                {
                    "mode": "rotation_only",
                    "pairs": extrinsics.pairs,
                    "residual_rotation_deg": extrinsics.residual_rotation_deg,
                    "axis_spread_deg": extrinsics.axis_spread_deg,
                    "t_bc_source": "measured",
                    "methods": extrinsics.methods,
                    "note": "R_bc 由姿态旋转估计，t_bc 使用尺量值",
                }
                if extrinsics is not None
                else None
            ),
            "source_dir": str(flight_dir),
            "translation_diagnostic": dict(translation_diagnostic),
            **({"telemetry_quality": dict(telemetry_quality)} if telemetry_quality else {}),
        },
    }
