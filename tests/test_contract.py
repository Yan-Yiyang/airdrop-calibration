"""离线标定包的独立文件格式、导入边界与几何契约。"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_calibration_package_imports_without_airdrop_from_other_cwd(tmp_path: Path) -> None:
    """子解释器阻断运行包导入，保证工具与默认 cwd 解耦。"""
    script = """
import importlib.abc
import sys

class BlockAirdrop(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'airdrop' or fullname.startswith('airdrop.'):
            raise AssertionError('标定工具不应导入 airdrop: ' + fullname)
        return None

sys.meta_path.insert(0, BlockAirdrop())
from airdrop_calibration import calibrate
import airdrop_calibration.algorithms
import airdrop_calibration.flight_log
import airdrop_calibration.report
assert calibrate.rotation_z(90).shape == (3, 3)
assert not any(name == 'airdrop' or name.startswith('airdrop.') for name in sys.modules)
assert 'cv2' not in sys.modules
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_calibration_python_sources_have_no_airdrop_imports() -> None:
    """静态检查惰性或条件分支里也没有运行包导入。"""
    import ast

    paths = [
        REPO_ROOT / "airdrop_calibration" / "calibrate.py",
        *sorted((REPO_ROOT / "airdrop_calibration").glob("*.py")),
    ]
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = [
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        ]
        imports.extend(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert not any(name == "airdrop" or name.startswith("airdrop.") for name in imports), path


def test_frame_reader_rejects_paths_outside_flight_directory(tmp_path: Path) -> None:
    """帧索引中的路径不能借 ../ 访问飞行目录之外的文件。"""
    from airdrop_calibration.flight_log import load_flight_frames_paired

    flight = tmp_path / "flight"
    flight.mkdir()
    (tmp_path / "outside.jpg").write_bytes(b"not used")
    record = {
        "filename": "../outside.jpg",
        "capture_timestamp": 1.0,
        "received_timestamp": 1.1,
    }
    (flight / "frames_index.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="越过 frames 目录"):
        load_flight_frames_paired(flight)


def test_frame_reader_rejects_absolute_paths_even_inside_frames(tmp_path: Path) -> None:
    """相对路径契约在 Windows 盘符、UNC 与目录内绝对路径下都一致。"""
    from airdrop_calibration.flight_log import load_flight_frames_paired

    flight = tmp_path / "flight"
    frames = flight / "frames"
    frames.mkdir(parents=True)
    for filename in (
        str((frames / "inside.jpg").resolve()),
        "C:\\archive\\image.jpg",
        "\\\\server\\share\\image.jpg",
    ):
        record = {
            "filename": filename,
            "capture_timestamp": 1.0,
            "received_timestamp": 1.1,
        }
        (flight / "frames_index.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="相对路径"):
            load_flight_frames_paired(flight)


def test_frame_reader_rejects_nonfinite_timestamps(tmp_path: Path) -> None:
    """无效的帧时间戳不能进入标定时间轴。"""
    from airdrop_calibration.flight_log import load_flight_stamps

    flight = tmp_path / "flight"
    flight.mkdir()
    record = {
        "filename": "frame.jpg",
        "capture_timestamp": 1.0,
        "received_timestamp": float("nan"),
    }
    (flight / "frames_index.jsonl").write_text(
        json.dumps(record, allow_nan=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="received_timestamp.*有限数值"):
        load_flight_stamps(flight)


def test_main_uses_explicit_paths_from_an_unrelated_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """入口只按调用方给定路径工作，不从当前目录猜仓库根目录。"""
    from airdrop_calibration import calibrate

    flight = tmp_path / "input" / "flight"
    flight.mkdir(parents=True)
    outside = tmp_path / "unrelated"
    outside.mkdir()
    output = tmp_path / "result" / "camera_calib.json"
    calls: list[tuple[Path, str]] = []

    def fake_calibrate(flight_dir: Path, **kwargs: object) -> dict[str, object]:
        calls.append((flight_dir, str(kwargs["source"])))
        return {"camera_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]}

    monkeypatch.setattr(calibrate, "calibrate", fake_calibrate)
    monkeypatch.chdir(outside)
    assert calibrate.main(flight_dir=flight, output_path=output) == 0
    assert calls == [(flight, "measured_calibration")]
    assert json.loads(output.read_text(encoding="utf-8"))["camera_matrix"][0][0] == 1.0


def test_known_handeye_and_received_lag_contract() -> None:
    """已知 SE(3) 手眼解与 received 时间轴延时保持原有数值约定。"""
    from airdrop_calibration.algorithms import (
        measure_frame_telemetry_lag,
        rotation_x,
        rotation_y,
        rotation_z,
        solve_hand_eye,
    )
    from airdrop_calibration.models import Telemetry

    true_x = np.eye(4)
    true_x[:3, :3] = rotation_z(37.0) @ rotation_y(-12.0) @ rotation_x(8.0)
    true_x[:3, 3] = (0.12, -0.04, 0.2)
    bodies = []
    for index in range(8):
        body = np.eye(4)
        body[:3, :3] = rotation_z(index * 11.0) @ rotation_x((index % 3) * 9.0)
        body[:3, 3] = (index * 0.03, (index % 3) * 0.02, -index * 0.01)
        bodies.append(body)
    cameras = [np.linalg.inv(true_x) @ np.linalg.inv(body) for body in bodies]
    a_list = [np.linalg.inv(bodies[j]) @ bodies[i] for i in range(8) for j in range(i + 1, 8)]
    b_list = [cameras[j] @ np.linalg.inv(cameras[i]) for i in range(8) for j in range(i + 1, 8)]
    rotation, translation, _, _ = solve_hand_eye(a_list, b_list)
    assert translation is not None
    assert np.allclose(rotation, true_x[:3, :3], atol=1e-8)
    assert np.allclose(translation, true_x[:3, 3], atol=1e-8)

    count = 90
    physical = [100.0 + index / 30.0 for index in range(count)]
    video_delay, telemetry_delay = 0.12, 0.04
    angles = [
        24.0 * math.sin(index * 0.13) + 8.0 * math.sin(index * 0.037) for index in range(count)
    ]
    frame_rotations = [rotation_z(angle) for angle in angles]
    snapshots = [
        Telemetry(
            timestamp=stamp + telemetry_delay,
            quaternion_w=math.cos(math.radians(angle) / 2.0),
            quaternion_x=0.0,
            quaternion_y=0.0,
            quaternion_z=math.sin(math.radians(angle) / 2.0),
        )
        for stamp, angle in zip(physical, angles, strict=True)
    ]
    received = [stamp + video_delay for stamp in physical]
    measurement = measure_frame_telemetry_lag(
        received,
        [stamp - 0.2 for stamp in received],
        frame_rotations,
        snapshots,
        baseline_s=0.2,
        search_s=0.25,
        step_s=1.0 / 30.0,
    )
    assert measurement.absolute.lag_s == pytest.approx(video_delay - telemetry_delay, abs=0.04)
    assert measurement.time_base == "received"
