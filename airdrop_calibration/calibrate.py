"""相机标定三步工作流与命令行配置。"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .algorithms import (
    align_body_poses,
    board_object_points,
    detect_corners,
    estimate_translation_fixed_rotation,
    measure_frame_telemetry_lag,
    run_extrinsics,
    run_intrinsics_from_corners,
    solve_board_pose,
)
from .flight_log import _frame_path, iter_flight_frames_paired, load_flight_telemetry
from .models import ExtrinsicsResult, FrameObservation, LagMeasurement, LagResult, Telemetry
from .report import CalibrationSource, build_calibration_report
from .settings import (
    ESTIMATE_TRANSLATION,
    FLIGHT_DIR,
    LAG_CROSS_CHECK_TOL_S,
    LAG_SEARCH_S,
    LAG_STEP_S,
    MAX_FRAMES,
    MAX_MOTION_PAIRS,
    MEASURED_T_BC,
    OUTPUT_PATH,
    PATTERN_SIZE,
    SQUARE_SIZE_M,
    STRICT,
)
from .source_timeline import SourceTimeline

LOGGER = logging.getLogger("calibrate")


def _translation_diagnostic(
    *,
    status: str,
    reason: str | None,
    estimated: np.ndarray | None,
    residual: float | None,
    pairs: int,
    measured: Sequence[float],
) -> dict[str, Any]:
    measured_values = [float(value) for value in measured]
    estimate_values = None if estimated is None else estimated.astype(float).tolist()
    delta = None if estimated is None else estimated - np.asarray(measured_values)
    return {
        "status": status,
        "reason": reason,
        "measured": measured_values,
        "estimated": estimate_values,
        "delta_xyz_m": None if delta is None else delta.tolist(),
        "delta_m": None if delta is None else float(np.linalg.norm(delta)),
        "residual_m": residual,
        "pairs": pairs,
    }


def _log_translation_diagnostic(diagnostic: dict[str, Any]) -> None:
    if diagnostic["estimated"] is None:
        LOGGER.info(
            "  平移对比%s", f"跳过：{diagnostic['reason']}" if diagnostic["reason"] else "已关闭"
        )
        return
    LOGGER.info(
        "  平移对比（机体 FRD，m）：尺量 %s，估计 %s，逐轴差 %s，距离差 %.4f m",
        diagnostic["measured"],
        diagnostic["estimated"],
        diagnostic["delta_xyz_m"],
        diagnostic["delta_m"],
    )


def calibrate(  # noqa: PLR0912 - 三步流水线保留清晰的失败分支
    flight_dir: Path,
    *,
    pattern_size: tuple[int, int] = PATTERN_SIZE,
    square_size: float = SQUARE_SIZE_M,
    lag_search_s: float = LAG_SEARCH_S,
    lag_step_s: float = LAG_STEP_S,
    max_frames: int | None = MAX_FRAMES,
    source: CalibrationSource = "measured_calibration",
    estimate_translation: bool = ESTIMATE_TRANSLATION,
) -> dict[str, Any]:
    """跑完三步，返回可写入 ``camera_calib.json`` 的字典。"""
    import cv2  # 惰性导入：重依赖，只在真正做标定时加载（见模块 docstring）

    if max_frames is not None and (isinstance(max_frames, bool) or max_frames < 1):
        raise ValueError("max_frames 必须是正整数或 None")
    telemetry_rows = load_flight_telemetry(flight_dir)
    capture_quality = _validate_full_update_capture(flight_dir, telemetry_rows)
    LOGGER.info("读取标定素材: %s", flight_dir)
    observations: list[FrameObservation] = []
    frame_count = 0
    first_received: float | None = None
    last_received: float | None = None
    image_size: tuple[int, int] | None = None
    first_lag: float | None = None
    min_lag = math.inf
    max_lag = -math.inf
    for stamp, filename, image in iter_flight_frames_paired(flight_dir, max_frames):
        frame_count += 1
        first_received = stamp.received_timestamp if first_received is None else first_received
        last_received = stamp.received_timestamp
        first_lag = stamp.lag_s if first_lag is None else first_lag
        min_lag = min(min_lag, stamp.lag_s)
        max_lag = max(max_lag, stamp.lag_s)
        current_size = (int(image.shape[1]), int(image.shape[0]))
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            raise ValueError(f"标定图像尺寸不一致: {image_size} 与 {current_size} ({filename})")
        corners = detect_corners(image, pattern_size)
        if corners is not None:
            digest = hashlib.sha256(memoryview(image).cast("B")).digest()
            observations.append(FrameObservation(stamp, filename, corners, current_size, digest))
        del corners, image
    if not frame_count:
        raise ValueError(f"{flight_dir} 里没有可用帧")
    assert first_lag is not None
    lag_baseline = first_lag
    if max_lag - min_lag > 1e-6:
        LOGGER.warning(
            "  素材里各帧记录的 lag 不一致（%.4f ~ %.4f）：互核基准取第一可读帧",
            min_lag,
            max_lag,
        )
    LOGGER.info(
        "分析 %d 帧（时间轴 = received；录制 lag 基准 %.4f s），其中 %d 帧有有效角点",
        frame_count,
        lag_baseline,
        len(observations),
    )

    LOGGER.info("步骤一：内参标定")
    intrinsics = run_intrinsics_from_corners(
        [(observation.image_size, observation.corners) for observation in observations],
        pattern_size=pattern_size,
        square_size=square_size,
    )
    LOGGER.info(
        "  视图 %d，重投影 RMS %.4f px，fx=%.1f fy=%.1f",
        intrinsics.views,
        intrinsics.rms,
        intrinsics.camera_matrix[0, 0],
        intrinsics.camera_matrix[1, 1],
    )

    LOGGER.info("步骤二：画面-遥测时间差")
    board_points = board_object_points(pattern_size, square_size)
    board_rotations: list[np.ndarray] = []
    pose_times: list[float] = []
    pose_times_capture: list[float] = []
    board_poses: list[tuple[float, np.ndarray, np.ndarray]] = []
    rejected = 0

    frames_dir = flight_dir / "frames"
    for observation in observations:
        stamp = observation.stamp
        filename = observation.filename
        path = _frame_path(frames_dir, filename)
        try:
            encoded = np.fromfile(str(path), dtype=np.uint8)
        except OSError as exc:
            raise ValueError(f"位姿分析第二遍无法读取帧: {filename}") from exc
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"位姿分析第二遍无法解码帧: {filename}")
        current_size = (int(image.shape[1]), int(image.shape[0]))
        if current_size != observation.image_size:
            raise ValueError(f"两遍分析间图像尺寸改变: {filename}")
        digest = hashlib.sha256(memoryview(image).cast("B")).digest()
        if digest != observation.image_digest:
            raise ValueError(f"两遍分析间图像像素内容改变: {filename}")
        pose = solve_board_pose(
            image,
            observation.corners,
            board_points,
            intrinsics.camera_matrix,
            intrinsics.dist_coeffs,
            pattern_size=pattern_size,
            square_size=square_size,
        )
        if pose is None:
            rejected += 1
            del image, encoded
            continue
        rvec, tvec = pose
        rotation, _ = cv2.Rodrigues(rvec)
        board_rotations.append(rotation)
        pose_times.append(stamp.received_timestamp)
        pose_times_capture.append(stamp.capture_timestamp)
        board_poses.append((stamp.received_timestamp, rvec, tvec))
        del image, encoded
    del observations
    if rejected:
        LOGGER.warning("  丢弃 %d 帧（棋盘方向或相位判据未通过）", rejected)
    # BUG：合并快照 timestamp 不是四元数采样/接收时刻，会把其它字段更新延迟混入标定。
    timeline = SourceTimeline(telemetry_rows, include_position=False)
    telemetry_quality: dict[str, Any] = {"source_timeline": timeline.quality_meta}
    if capture_quality is not None:
        telemetry_quality["capture"] = capture_quality
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
    translation_status = "disabled" if not estimate_translation else "skipped"
    translation_reason = "disabled_by_user" if not estimate_translation else "not_run"
    translation_estimate: np.ndarray | None = None
    translation_residual: float | None = None
    translation_pairs = 0

    if board_poses and len(snapshots) >= 2 and lag.ok:
        paired_board, paired_body = align_body_poses(
            board_poses, snapshots, lag_s=lag.lag_s, pose_at=timeline.rotation_at
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
            if estimate_translation:
                try:
                    position_timeline = SourceTimeline(telemetry_rows)
                    position_board: list[tuple[float, np.ndarray, np.ndarray]] = []
                    position_body: list[np.ndarray] = []
                    for board_pose in paired_board:
                        body = position_timeline.pose_at(board_pose[0] - lag.lag_s)
                        if body is not None:
                            position_board.append(board_pose)
                            position_body.append(body)
                    (
                        translation_estimate,
                        translation_residual,
                        translation_pairs,
                    ) = estimate_translation_fixed_rotation(
                        position_board,
                        position_body,
                        extrinsics.r_bc,
                        max_pairs=MAX_MOTION_PAIRS,
                    )
                    translation_status, translation_reason = "estimated", None
                except (ValueError, np.linalg.LinAlgError) as exc:
                    translation_reason = str(exc)
        except ValueError as exc:
            LOGGER.warning("  手眼标定失败: %s", exc)
            if estimate_translation:
                translation_reason = f"rotation_estimation_failed: {exc}"
    else:
        LOGGER.warning("  缺少棋盘格/遥测位姿或有效时间差，跳过外参")
        if estimate_translation:
            translation_reason = "rotation_or_time_alignment_unavailable"

    diagnostic = _translation_diagnostic(
        status=translation_status,
        reason=translation_reason,
        estimated=translation_estimate,
        residual=translation_residual,
        pairs=translation_pairs,
        measured=MEASURED_T_BC,
    )
    _log_translation_diagnostic(diagnostic)

    report = build_calibration_report(
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
        telemetry_quality=telemetry_quality,
        translation_diagnostic=diagnostic,
    )
    report["meta"]["frame_analysis"] = {
        "recorded_frames": capture_quality["frame_count"] if capture_quality else None,
        "analyzed_frames": frame_count,
        "received_time_range_s": [first_received, last_received],
        "intrinsic_views": intrinsics.views,
        "pnp_poses": len(board_poses),
        "max_frames": max_frames,
    }
    return report


def _validate_full_update_capture(
    flight_dir: Path, telemetry: Sequence[Telemetry]
) -> dict[str, Any] | None:
    """对标定 profile 的记录做 fail-closed 完整性核验。"""
    config_path = flight_dir / "config_snapshot.json"
    if not config_path.is_file():
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("record", {}).get("telemetry_hz") != 0:
        return None

    event_path = flight_dir / "events.jsonl"
    events = [
        json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines() if line
    ]
    if not events or events[-1].get("kind") != "recorder_stopped":
        raise ValueError("标定全量记录缺少 recorder_stopped 收尾事件")
    stopped = events[-1]
    counts = {
        name: stopped.get(name)
        for name in (
            "telemetry_received",
            "telemetry_enqueued",
            "telemetry_written",
            "telemetry_dropped",
            "telemetry_write_errors",
            "telemetry_pending",
            "frame_encode_errors",
            "finalize_errors",
        )
    }
    if (
        counts["telemetry_received"] != counts["telemetry_enqueued"]
        or counts["telemetry_enqueued"] != counts["telemetry_written"]
        or counts["telemetry_dropped"] != 0
        or counts["telemetry_write_errors"] != 0
        or counts["telemetry_pending"] != 0
        or counts["frame_encode_errors"] != 0
        or counts["finalize_errors"] != 0
    ):
        raise ValueError(f"标定全量遥测队列不完整: {counts}")
    if stopped.get("frames_skipped") != 0:
        raise ValueError(f"标定全量记录有帧丢失: frames_skipped={stopped['frames_skipped']}")
    if stopped.get("events") != len(events) - 1:
        raise ValueError("标定 events.jsonl 行数与 recorder_stopped 统计不一致")

    telemetry_path = flight_dir / "telemetry.jsonl"
    with telemetry_path.open(encoding="utf-8") as telemetry_file:
        raw_lines = sum(bool(line.strip()) for line in telemetry_file)
    if raw_lines != counts["telemetry_written"] or len(telemetry) != raw_lines:
        raise ValueError(
            "标定 telemetry.jsonl 行数、有效 DTO 数与 writer 统计不一致: "
            f"lines={raw_lines}, dto={len(telemetry)}, written={counts['telemetry_written']}"
        )

    frames_path = flight_dir / "frames_index.jsonl"
    import cv2  # 惰性导入：完整性检查只在离线标定时加载 OpenCV

    frame_count = 0
    first_frame_time: float | None = None
    last_frame_time: float | None = None
    frames_dir = flight_dir / "frames"
    with frames_path.open(encoding="utf-8") as frame_file:
        for line_number, line in enumerate(frame_file, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                timestamp = float(item["received_timestamp"])
                filename = str(item["filename"])
                expected_bytes = int(item["bytes"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"标定帧索引第 {line_number} 行缺少有效时间戳、文件名或字节数"
                ) from exc
            if not math.isfinite(timestamp) or (
                last_frame_time is not None and timestamp <= last_frame_time
            ):
                raise ValueError("标定帧接收时间戳必须有限且严格递增")
            try:
                path = _frame_path(frames_dir, filename)
            except ValueError as exc:
                raise ValueError(f"标定帧索引包含非法文件名: {filename}") from exc
            try:
                encoded = path.read_bytes()
            except OSError as exc:
                raise ValueError(f"标定帧文件缺失或不可读: {filename}") from exc
            image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
            if len(encoded) != expected_bytes or image is None:
                raise ValueError(f"标定帧文件长度不符或无法解码: {filename}")
            del image, encoded
            first_frame_time = timestamp if first_frame_time is None else first_frame_time
            last_frame_time = timestamp
            frame_count += 1
    if frame_count != stopped.get("frames_written") or frame_count < 2:
        raise ValueError("标定帧索引与 recorder_stopped 统计不一致，或有效帧不足")
    assert first_frame_time is not None and last_frame_time is not None
    span = last_frame_time - first_frame_time
    return {
        "scope": "process_accepted_samples",
        "telemetry": counts,
        "frame_count": frame_count,
        "frame_hz": (frame_count - 1) / span,
    }


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
    #: 可选的调试前缀帧数上限；默认分析完整采集
    max_frames: int | None = MAX_FRAMES
    #: 严格模式：外参/时间差没标出来时返回非 0（默认只警告，照样写文件）
    strict: bool = STRICT
    source: CalibrationSource = "measured_calibration"
    estimate_translation: bool = ESTIMATE_TRANSLATION


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
        estimate_translation=settings.estimate_translation,
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
        problems.append("手眼外参没有标出来（R_bc=null）")
    if problems:
        for item in problems:
            LOGGER.error("严格模式判定标定不合格：%s", item)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
