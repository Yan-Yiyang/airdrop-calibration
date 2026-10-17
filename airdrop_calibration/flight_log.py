"""读取 FlightRecorder JSONL 与帧文件，不依赖运行时包。"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np

from .models import FrameStamp, Telemetry
from .settings import TIME_BASE_CAPTURE, TIME_BASE_RECEIVED

LOGGER = logging.getLogger("calibrate")


def _stamp_value(stamp: FrameStamp, time_base: str) -> float:
    """按时间轴取值；未知时间轴显式报错。"""
    if time_base == TIME_BASE_RECEIVED:
        return stamp.received_timestamp
    if time_base == TIME_BASE_CAPTURE:
        return stamp.capture_timestamp
    raise ValueError(
        f"未知的 time_base: {time_base!r}（可选 {TIME_BASE_RECEIVED!r} / {TIME_BASE_CAPTURE!r}）"
    )


def _read_flight_index(flight_dir: Path, limit: int | None) -> list[tuple[FrameStamp, str]]:
    """一次遍历读取帧时间与相对文件名；标定必须有 received 时间戳。"""
    index_path = flight_dir / "frames_index.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(f"缺少帧索引: {index_path}")
    entries: list[tuple[FrameStamp, str]] = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"帧索引记录必须是对象（{index_path} 第 {len(entries) + 1} 行）")
        if "received_timestamp" not in record:
            raise ValueError(
                f"帧索引缺少 received_timestamp（{index_path} 第 {len(entries) + 1} 行）："
                "标定第 2/3 步要用收到时刻这个原始量，不能拿 capture_timestamp 顶替。"
                "旧素材可显式用 load_flight_index(..., time_base='capture') 诊断。"
            )
        stamp = FrameStamp(
            received_timestamp=_finite_number(record["received_timestamp"], "received_timestamp"),
            capture_timestamp=_finite_number(record["capture_timestamp"], "capture_timestamp"),
            lag_s=_finite_number(record.get("lag", 0.0), "lag"),
        )
        entries.append((stamp, str(record["filename"])))
        if limit is not None and len(entries) >= limit:
            break
    return entries


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是有限数值")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} 必须是有限数值")
    return number


def load_flight_stamps(flight_dir: Path, limit: int | None = None) -> list[FrameStamp]:
    """读取帧索引的 received/capture/lag 三项时间记录。"""
    return [stamp for stamp, _ in _read_flight_index(flight_dir, limit)]


def load_flight_index(
    flight_dir: Path,
    limit: int | None = None,
    *,
    time_base: str = TIME_BASE_RECEIVED,
) -> list[tuple[float, str]]:
    """读取 ``[(指定时间戳, 相对帧文件名), ...]``。"""
    return [
        (_stamp_value(stamp, time_base), name)
        for stamp, name in _read_flight_index(flight_dir, limit)
    ]


def _frame_path(frames_dir: Path, name: str) -> Path:
    """解析帧路径并拒绝绝对路径、目录穿越和越界符号链接。"""
    windows_name = PureWindowsPath(name)
    if Path(name).is_absolute() or windows_name.drive or windows_name.root:
        raise ValueError(f"帧文件名必须是 frames 内的相对路径: {name!r}")
    flight_root = frames_dir.parent.resolve()
    root = frames_dir.resolve()
    if not root.is_relative_to(flight_root):
        raise ValueError("frames 目录路径越过飞行目录")
    candidate = (frames_dir / name).resolve()
    if not candidate.is_relative_to(root) or candidate == root:
        raise ValueError(f"帧文件路径越过 frames 目录: {name!r}")
    return candidate


def load_flight_frames_paired(
    flight_dir: Path, limit: int = 200
) -> list[tuple[FrameStamp, np.ndarray]]:
    """读取帧并与产生它的同一索引记录配对；缺失或坏图像连同时间一起跳过。"""
    import cv2

    frames_dir = flight_dir / "frames"
    paired: list[tuple[FrameStamp, np.ndarray]] = []
    skipped = 0
    for stamp, name in _read_flight_index(flight_dir, None):
        path = _frame_path(frames_dir, name)
        if not path.is_file():
            skipped += 1
            continue
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            skipped += 1
            continue
        paired.append((stamp, image))
        if len(paired) >= limit:
            break
    if skipped:
        LOGGER.warning(
            "帧索引里有 %d 帧缺失或解不出（已连同其时间记录一起跳过）",
            skipped,
        )
    return paired


def load_flight_telemetry(flight_dir: Path) -> list[Telemetry]:
    """读遥测 JSONL；坏行跳过，缺失或非有限时间戳的行不进入时间轴。"""
    path = flight_dir / "telemetry.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"缺少遥测日志: {path}")
    expected = set(Telemetry.__dataclass_fields__)
    snapshots: list[Telemetry] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                LOGGER.warning("telemetry.jsonl 第 %d 行不是合法 JSON，跳过", line_no)
                continue
            if not isinstance(data, dict):
                LOGGER.warning("telemetry.jsonl 第 %d 行不是对象，跳过", line_no)
                continue
            if "timestamp" not in data:
                LOGGER.warning("telemetry.jsonl 第 %d 行缺少 timestamp，跳过", line_no)
                continue
            known = {key: value for key, value in data.items() if key in expected}
            try:
                snapshots.append(Telemetry(**known))
            except (TypeError, ValueError, OverflowError) as exc:
                LOGGER.warning("telemetry.jsonl 第 %d 行字段异常，跳过: %s", line_no, exc)
    return sorted(snapshots, key=lambda snapshot: snapshot.timestamp)


def load_flight_timestamps(
    flight_dir: Path,
    limit: int = 200,
    *,
    time_base: str = TIME_BASE_RECEIVED,
) -> list[float]:
    """读取帧索引时间；标定默认使用收到时刻。"""
    return [stamp for stamp, _ in load_flight_index(flight_dir, limit, time_base=time_base)]
