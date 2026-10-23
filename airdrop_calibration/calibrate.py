"""相机标定兼容入口：保留配置、三步工作流与历史公开函数。"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .algorithms import (
    align_body_poses,
    angular_speed_from_poses,
    board_object_points,
    body_pose_matrix,
    detect_corners,
    estimate_lag_by_correlation,
    lever_arm_tolerance,
    look_at_board,
    measure_frame_telemetry_lag,
    pattern_phase_ok,
    quaternion_to_matrix_local,
    relative_transform,
    render_board,
    rotation_angle_deg,
    rotation_log,
    rotation_x,
    rotation_y,
    rotation_z,
    run_extrinsics,
    run_intrinsics,
    solve_board_poses,
    solve_hand_eye,
    telemetry_pose_at,
    telemetry_span_ok,
)
from .algorithms import lever_arm_report as _lever_arm_report
from .flight_log import (
    load_flight_frames_paired,
    load_flight_index,
    load_flight_stamps,
    load_flight_telemetry,
    load_flight_timestamps,
)
from .models import (
    ExtrinsicsResult,
    FrameStamp,
    IntrinsicsResult,
    LagMeasurement,
    LagResult,
    Telemetry,
    TelemetryLike,
)
from .report import (
    SCHEMA_VERSION,
    CalibrationSource,
    build_calibration_report,
)
from .settings import (
    BODY_SPAN_FLOOR_M,
    BODY_SPAN_RATIO_MAX,
    BODY_SPAN_RATIO_MIN,
    FLIGHT_DIR,
    LAG_CROSS_CHECK_TOL_S,
    LAG_SEARCH_S,
    LAG_STEP_S,
    LEVER_ARM_RESIDUAL_FLOOR_M,
    LEVER_ARM_RESIDUAL_RATIO,
    MAX_FRAMES,
    MAX_MOTION_PAIRS,
    MAX_VIEWS,
    MEASURED_T_BC,
    MIN_AXIS_SPREAD_DEG,
    MIN_PAIR_ROTATION_DEG,
    MIN_VIEWS,
    OUTPUT_PATH,
    PATTERN_SIZE,
    SQUARE_SIZE_M,
    STRICT,
    TIME_BASE_CAPTURE,
    TIME_BASE_RECEIVED,
)
from .source_timeline import SourceTimeline

LOGGER = logging.getLogger("calibrate")

__all__ = [
    "BODY_SPAN_FLOOR_M",
    "BODY_SPAN_RATIO_MAX",
    "BODY_SPAN_RATIO_MIN",
    "FLIGHT_DIR",
    "LAG_CROSS_CHECK_TOL_S",
    "LAG_SEARCH_S",
    "LAG_STEP_S",
    "LEVER_ARM_RESIDUAL_FLOOR_M",
    "LEVER_ARM_RESIDUAL_RATIO",
    "MAX_FRAMES",
    "MAX_MOTION_PAIRS",
    "MAX_VIEWS",
    "MEASURED_T_BC",
    "MIN_AXIS_SPREAD_DEG",
    "MIN_PAIR_ROTATION_DEG",
    "MIN_VIEWS",
    "OUTPUT_PATH",
    "PATTERN_SIZE",
    "SCHEMA_VERSION",
    "SQUARE_SIZE_M",
    "STRICT",
    "TIME_BASE_CAPTURE",
    "TIME_BASE_RECEIVED",
    "CalibrateConfig",
    "CalibrationSource",
    "ExtrinsicsResult",
    "FrameStamp",
    "IntrinsicsResult",
    "LagMeasurement",
    "LagResult",
    "Telemetry",
    "TelemetryLike",
    "align_body_poses",
    "angular_speed_from_poses",
    "board_object_points",
    "body_pose_matrix",
    "build_calibration_report",
    "build_config",
    "calibrate",
    "detect_corners",
    "estimate_lag_by_correlation",
    "lever_arm_report",
    "lever_arm_tolerance",
    "load_flight_frames_paired",
    "load_flight_index",
    "load_flight_stamps",
    "load_flight_telemetry",
    "load_flight_timestamps",
    "look_at_board",
    "main",
    "measure_frame_telemetry_lag",
    "pattern_phase_ok",
    "quaternion_to_matrix_local",
    "relative_transform",
    "render_board",
    "rotation_angle_deg",
    "rotation_log",
    "rotation_x",
    "rotation_y",
    "rotation_z",
    "run_extrinsics",
    "run_intrinsics",
    "solve_board_poses",
    "solve_hand_eye",
    "telemetry_pose_at",
    "telemetry_span_ok",
]


def lever_arm_report(
    result: ExtrinsicsResult | None,
    measured: Sequence[float] | None = None,
) -> dict[str, Any]:
    """报告尺量杆臂；默认值按调用时的兼容入口常量读取。"""
    return _lever_arm_report(result, MEASURED_T_BC if measured is None else measured)


def _log_lever_arm(arm: dict[str, Any]) -> None:
    """记录杆臂估计及其可信度判据。"""
    if arm["estimated"] is None:
        LOGGER.info("  杆臂按尺量: t_bc=%s（未估）", list(MEASURED_T_BC))
        return
    LOGGER.info(
        "  杆臂：尺量 %s，标定估计 %s，差 %.3f m",
        [round(value, 4) for value in arm["authoritative"]],
        [round(value, 4) for value in arm["estimated"]],
        arm["delta_m"],
    )
    if arm["estimated_reliable"]:
        LOGGER.info(
            "  估计值可采信（平移残差 %.4f m 在 %.4f m 以内；遥测位置跨度 %.2f m vs 画面 %.2f m）",
            arm["residual_translation_m"],
            arm["residual_tolerance_m"],
            arm["body_span_m"],
            arm["camera_span_m"],
        )
        return
    reasons = []
    if arm["residual_translation_m"] > arm["residual_tolerance_m"]:
        reasons.append(
            f"平移残差 {arm['residual_translation_m']:.4f} m 超出 "
            f"{arm['residual_tolerance_m']:.4f} m"
        )
    if not arm["body_span_ok"]:
        reasons.append(
            f"遥测位置跨度 {arm['body_span_m']:.3f} m 不合格（画面 {arm['camera_span_m']:.3f} m，"
            f"下限 {BODY_SPAN_FLOOR_M} m、比值需在 [{BODY_SPAN_RATIO_MIN}, {BODY_SPAN_RATIO_MAX}] 内）"
        )
    LOGGER.warning(
        "  估计值不可信（%s）——室内手持无 GPS 时遥测位置不可用，以尺量值 %s 为准；"
        "遥测位置几乎不动时残差也可能是干净的",
        "；".join(reasons) or "未知原因",
        list(MEASURED_T_BC),
    )


def calibrate(
    flight_dir: Path,
    *,
    pattern_size: tuple[int, int] = PATTERN_SIZE,
    square_size: float = SQUARE_SIZE_M,
    lag_search_s: float = LAG_SEARCH_S,
    lag_step_s: float = LAG_STEP_S,
    max_frames: int = MAX_FRAMES,
    source: CalibrationSource = "measured_calibration",
) -> dict[str, Any]:
    """跑完三步，返回可写入 ``camera_calib.json`` 的字典。"""
    import cv2  # 惰性导入：重依赖，只在真正做标定时加载（见模块 docstring）

    LOGGER.info("读取标定素材: %s", flight_dir)
    paired = load_flight_frames_paired(flight_dir, max_frames)
    if not paired:
        raise ValueError(f"{flight_dir} 里没有可用帧")
    # 帧与时间记录来自同一条索引记录，**结构上不可能错位**。
    # ⚠ 第 2/3 步一律用 received（原始"收到帧"时刻）：capture 是录制时按配置的 lag 假设出来的
    # 拍摄时刻，拿它测时间差只会测出"假设的残差"（见 load_flight_stamps 与
    # measure_frame_telemetry_lag 的说明）。capture 轴只用于互核。
    baselines = [stamp.lag_s for stamp, _ in paired]
    frames = [image for _, image in paired]
    lag_baseline = baselines[0]
    if max(baselines) - min(baselines) > 1e-6:
        LOGGER.warning(
            "  素材里各帧记录的 lag 不一致（%.4f ~ %.4f）：互核基准取第一帧",
            min(baselines),
            max(baselines),
        )
    LOGGER.info(
        "共 %d 帧（时间轴 = received 收到时刻；录制时的 lag 基准 %.4f s）",
        len(frames),
        lag_baseline,
    )

    # ---- 步骤一 ----
    LOGGER.info("步骤一：内参标定")
    intrinsics = run_intrinsics(frames, pattern_size=pattern_size, square_size=square_size)
    LOGGER.info(
        "  视图 %d，重投影 RMS %.4f px，fx=%.1f fy=%.1f",
        intrinsics.views,
        intrinsics.rms,
        intrinsics.camera_matrix[0, 0],
        intrinsics.camera_matrix[1, 1],
    )

    # ---- 步骤二 ----
    LOGGER.info("步骤二：画面-遥测时间差")
    board_points = board_object_points(pattern_size, square_size)
    board_rotations: list[np.ndarray] = []
    pose_times: list[float] = []
    pose_times_capture: list[float] = []
    board_poses: list[tuple[float, np.ndarray, np.ndarray]] = []
    rejected = 0
    # 用 solve_board_poses（含相邻帧连续性定支）而不是裸 solvePnP：
    # 检测器的角点顺序可能与我们枚举的对象点相反，单帧无法分辨（见其 docstring），
    # 而**混用两种标签**会让第三步残差飙到几十度。
    for stamp, pose in zip(
        [record for record, _ in paired],
        solve_board_poses(
            frames,
            board_points,
            intrinsics.camera_matrix,
            intrinsics.dist_coeffs,
            pattern_size=pattern_size,
            square_size=square_size,
        ),
        strict=True,
    ):
        if pose is None:
            rejected += 1
            continue
        rvec, tvec = pose
        rotation, _ = cv2.Rodrigues(rvec)
        board_rotations.append(rotation)
        pose_times.append(stamp.received_timestamp)  # 第 3 步配对也用 received 轴
        pose_times_capture.append(stamp.capture_timestamp)
        board_poses.append((stamp.received_timestamp, rvec, tvec))
    if rejected:
        LOGGER.warning("  丢弃 %d 帧（角点检不出）", rejected)

    # BUG：合并快照 timestamp 不是四元数采样/接收时刻，会把其它字段更新延迟混入标定。
    timeline = SourceTimeline(load_flight_telemetry(flight_dir))
    snapshots = timeline.attitudes
    measurement = LagMeasurement(
        absolute=LagResult(lag_s=0.0, peak=float("nan"), samples=0),
        residual=None,
        baseline_s=lag_baseline,
        cross_check_ok=False,
    )
    if len(board_rotations) >= 3 and len(snapshots) >= 3:
        measurement = measure_frame_telemetry_lag(
            pose_times,
            pose_times_capture,
            board_rotations,
            snapshots,
            baseline_s=lag_baseline,
            search_s=lag_search_s,
            step_s=lag_step_s,
        )
        lag = measurement.absolute
        LOGGER.info(
            "  Δt = %.4f s（received 轴 = **绝对对齐量**；相关系数 %.3f，样本 %d）",
            lag.lag_s,
            lag.peak,
            lag.samples,
        )
        if measurement.residual is not None:
            LOGGER.info(
                "    互核：capture 轴残差 %.4f s + 录制基准 %.4f s = %.4f s%s",
                measurement.residual.lag_s,
                lag_baseline,
                lag_baseline + measurement.residual.lag_s,
                "" if measurement.cross_check_ok else "  ⚠ 与绝对量对不上，见下",
            )
            if not measurement.cross_check_ok:
                LOGGER.warning(
                    "    两条时间轴互核不通过（|基准 + 残差 − 绝对量| > %.3f s）："
                    "素材索引里的 lag 与录制配置可能不一致；结果仍以 received 轴为准",
                    LAG_CROSS_CHECK_TOL_S,
                )
    else:
        lag = measurement.absolute
        LOGGER.warning("  棋盘格位姿或四元数不足，跳过时间差估计")

    # ---- 步骤三 ----
    LOGGER.info("步骤三：手眼外参")
    extrinsics: ExtrinsicsResult | None = None
    if board_poses and len(snapshots) >= 2 and lag.ok:
        # 与画面**同时刻**的机体位姿：遥测 10 Hz、画面 30 Hz，
        # 直接把"第 i 条遥测"配给"第 i 帧"是错的，必须按帧时刻重采样
        paired_board, paired_body = align_body_poses(
            board_poses, snapshots, lag_s=lag.lag_s, pose_at=timeline.pose_at
        )
        dropped = len(board_poses) - len(paired_board)
        if dropped:
            LOGGER.info("  丢弃 %d 个时刻不在遥测覆盖内的棋盘格位姿", dropped)
        try:
            extrinsics = run_extrinsics(board_poses=paired_board, body_poses=paired_body)
            LOGGER.info(
                "  运动对 %d，旋转残差 %.3f°，旋转轴张角 %.1f°",
                extrinsics.pairs,
                extrinsics.residual_rotation_deg,
                extrinsics.axis_spread_deg,
            )
            # 杆臂：**尺量为准**（写进标定文件的是它），标定估计值只用来对表
            _log_lever_arm(lever_arm_report(extrinsics, measured=MEASURED_T_BC))
        except ValueError as exc:
            LOGGER.warning("  手眼标定失败: %s", exc)
    else:
        LOGGER.warning("  缺少棋盘格/遥测位姿或有效时间差，跳过外参")

    return build_calibration_report(
        intrinsics=intrinsics,
        lag=lag,
        measurement=measurement,
        extrinsics=extrinsics,
        flight_dir=Path(flight_dir),
        pattern_size=pattern_size,
        square_size=square_size,
        lag_search_s=lag_search_s,
        lag_step_s=lag_step_s,
        measured_t_bc=MEASURED_T_BC,
        source=source,
    )


@dataclass(frozen=True, slots=True)
class CalibrateConfig:
    """一次标定用到的全部参数（默认值 = 本文件顶部的常量）。"""

    #: 标定采集目录（FlightRecorder 产出）
    flight_dir: Path = FLIGHT_DIR
    output_path: Path = OUTPUT_PATH
    #: 棋盘格内角点数（列, 行）与方格边长（米）
    pattern_size: tuple[int, int] = PATTERN_SIZE
    square_size: float = SQUARE_SIZE_M
    #: 时间差搜索范围与步长（秒）
    lag_search_s: float = LAG_SEARCH_S
    lag_step_s: float = LAG_STEP_S
    #: 最多读取多少帧
    max_frames: int = MAX_FRAMES
    #: 严格模式：外参/时间差没标出来时返回非 0（默认只警告，照样写文件）
    strict: bool = STRICT
    source: CalibrationSource = "measured_calibration"


def build_config(**overrides) -> CalibrateConfig:
    """按关键字覆盖派生一份标定配置（``dataclasses.replace``；未知字段直接报错）。"""
    return replace(CalibrateConfig(), **overrides)


def main(**overrides) -> int:
    """跑一次三步标定并写出 JSON；关键字与 :func:`build_config` 一致。

    退出码：0 = 正常写出；2 = ``strict=True`` 且外参或时间差没标出来；1 = 出错（异常抛出）。
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    settings = build_config(**overrides)
    flight_dir = Path(settings.flight_dir)
    if not flight_dir.is_dir():
        # 失败要显式：目录不存在直接抛，不静默换个目录
        raise FileNotFoundError(f"飞行目录不存在: {flight_dir}")

    payload = calibrate(
        flight_dir,
        pattern_size=settings.pattern_size,
        square_size=settings.square_size,
        lag_search_s=settings.lag_search_s,
        lag_step_s=settings.lag_step_s,
        max_frames=settings.max_frames,
        source=settings.source,
    )
    out_path = Path(settings.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("标定结果已写入 %s", out_path.resolve())
    if not settings.strict:
        return 0
    # 严格模式：把"写进文件的那两处降级"当失败报出来（默认只警告）
    problems: list[str] = []
    if payload.get("telemetry_lag") is None:
        problems.append("画面-遥测时间差没有标出来（telemetry_lag=null）")
    if payload["meta"].get("extrinsics") is None:
        problems.append("手眼外参没有标出来（R_bc 用的是默认值）")
    if problems:
        for item in problems:
            LOGGER.error("严格模式判定标定不合格：%s", item)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
