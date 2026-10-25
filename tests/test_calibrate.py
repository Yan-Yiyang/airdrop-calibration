"""tools/calibrate.py 标定流水线的合成自洽验证（P6 验收口径之一）。

做法：用已知内参 / 已知 Δt / 已知 R_bc 生成合成棋盘格视图与机体姿态，
再让流水线去把它们解回来，逐项比对。

链路顺序（与真实采集一致）
--------------------------
相机（棋盘格在相机系下的位姿）由"机体在世界里的位姿 + 真值 X=R_bc/t_bc"导出，
因此流水线拿到的像素数据里天然含有真值外参——解不回来说明算法或约定错了。
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import airdrop_calibration.calibrate as calibrate_module
from airdrop_calibration.algorithms import (
    align_body_poses,
    angular_speed_from_poses,
    board_object_points,
    detect_corners,
    estimate_lag_by_correlation,
    estimate_translation_fixed_rotation,
    look_at_board,
    measure_frame_telemetry_lag,
    pattern_phase_ok,
    quaternion_to_matrix_local,
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
)
from airdrop_calibration.calibrate import calibrate
from airdrop_calibration.models import Telemetry as TelemetrySnapshot
from airdrop_calibration.settings import MEASURED_T_BC

# 真值
TRUE_FX, TRUE_FY = 900.0, 902.0
TRUE_CX, TRUE_CY = 640.0, 360.0
TRUE_SIZE = (1280, 720)
TRUE_DIST = np.array([-0.12, 0.03, 0.0005, -0.0007, 0.0])
PATTERN = (7, 5)
ORIENTABLE_PATTERN = (8, 5)
SQUARE = 0.03


def _bodies_with_real_translation(seed: int, count: int = 14) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    bodies = []
    for rotation in _random_rotations(count, seed):
        body = np.eye(4)
        body[:3, :3] = rotation
        body[:3, 3] = rng.normal(scale=0.3, size=3)
        bodies.append(body)
    return bodies


def _board_poses_from_cameras(
    cameras: Sequence[np.ndarray],
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    return [
        (float(index), cv2.Rodrigues(pose[:3, :3])[0], pose[:3, 3])
        for index, pose in enumerate(cameras)
    ]


def test_fixed_rotation_translation_recovers_truth_and_rejects_rank_deficiency() -> None:
    rotation = rotation_z(23.0) @ rotation_x(-17.0)
    truth = np.array([0.12, -0.04, 0.08])
    x_true = np.eye(4)
    x_true[:3, :3] = rotation
    x_true[:3, 3] = truth
    board_world = np.eye(4)
    board_world[:3, :3] = rotation_y(19.0)
    board_world[:3, 3] = [2.0, -1.0, 0.5]
    bodies = _bodies_with_real_translation(27)
    board = [np.linalg.inv(x_true) @ np.linalg.inv(body) @ board_world for body in bodies]
    poses = _board_poses_from_cameras(board)
    estimate, _, pairs = estimate_translation_fixed_rotation(poses, bodies, rotation)
    assert estimate == pytest.approx(truth, abs=1e-9)
    assert pairs > 0

    yaw_bodies = []
    for index in range(6):
        body = np.eye(4)
        body[:3, :3] = rotation_z(index * 20.0)
        body[:3, 3] = [index * 0.1, 0.0, 0.0]
        yaw_bodies.append(body)
    yaw_board = [np.linalg.inv(x_true) @ np.linalg.inv(body) @ board_world for body in yaw_bodies]
    with pytest.raises(ValueError, match="rank_deficient"):
        estimate_translation_fixed_rotation(
            _board_poses_from_cameras(yaw_board), yaw_bodies, rotation
        )


def test_fixed_rotation_translation_accepts_two_small_angle_pairs() -> None:
    rotation = rotation_z(18.0) @ rotation_y(-7.0)
    truth = np.array([0.12, -0.04, 0.08])
    x_true = np.eye(4)
    x_true[:3, :3] = rotation
    x_true[:3, 3] = truth
    board_world = np.eye(4)
    board_world[:3, 3] = [1.5, -0.7, 0.4]
    bodies = []
    for index, attitude in enumerate((np.eye(3), rotation_x(2.0), rotation_y(3.0))):
        body = np.eye(4)
        body[:3, :3] = attitude
        body[:3, 3] = [0.2 * index, -0.1 * index, 0.05 * index]
        bodies.append(body)
    board = [np.linalg.inv(x_true) @ np.linalg.inv(body) @ board_world for body in bodies]
    estimate, _, pairs = estimate_translation_fixed_rotation(
        _board_poses_from_cameras(board), bodies, rotation, max_pairs=2
    )
    assert pairs == 2
    assert estimate == pytest.approx(truth, abs=1e-9)


def test_invalid_position_fields_preserve_valid_quaternion() -> None:
    snapshot = TelemetrySnapshot(
        timestamp=1.0,
        quaternion_w=1.0,
        quaternion_x=0.0,
        quaternion_y=0.0,
        quaternion_z=0.0,
        quaternion_timestamp_us=1_000_000,
        quaternion_received_at=1.0,
        north_m=float("nan"),
        east_m=0.0,
        down_m=0.0,
        position_timestamp_us=cast(int, "invalid"),
    )
    assert snapshot.is_valid()
    assert snapshot.quaternion_w == 1.0
    assert snapshot.north_m is None
    assert snapshot.position_timestamp_us is None


def true_camera_matrix() -> np.ndarray:
    return np.array([[TRUE_FX, 0.0, TRUE_CX], [0.0, TRUE_FY, TRUE_CY], [0.0, 0.0, 1.0]])


def true_r_bc() -> np.ndarray:
    """相机→机体：绕机体 z +90°，再叠一点真实的安装倾斜。"""
    return rotation_z(90.0) @ rotation_y(-3.0) @ rotation_x(2.0)


TRUE_T_BC = np.array([0.06, -0.03, 0.09])


def _board_pose_in_camera(
    body_rotation: np.ndarray,
    body_position: np.ndarray,
    r_bc: np.ndarray,
    t_bc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """给定机体位姿与真值外参，返回棋盘格在相机系下的 (rvec, tvec)。

    棋盘格固定在世界原点（板面 z=0），机体位姿在世界系里给。
    ``T_cam_board = X^{-1} · T_body_world``，其中 ``X = [R_bc | t_bc]``。
    """
    body_world = np.eye(4)
    body_world[:3, :3] = body_rotation
    body_world[:3, 3] = body_position
    x = np.eye(4)
    x[:3, :3] = r_bc
    x[:3, 3] = t_bc
    camera = np.linalg.inv(x) @ body_world
    rvec, _ = cv2.Rodrigues(camera[:3, :3])
    return rvec, camera[:3, 3]


def _render_views(count: int, distance: float = 0.55) -> tuple[list[np.ndarray], list]:
    """渲染一批姿态多样的棋盘格视图（用真值内参/畸变），返回 (帧, 相机位姿)。"""
    frames: list[np.ndarray] = []
    poses: list[tuple[np.ndarray, np.ndarray]] = []
    for index in range(count):
        rvec, tvec = look_at_board(
            distance,
            true_camera_matrix(),
            tilt_deg=14.0 * math.sin(index * 0.9),
            yaw_deg=11.0 * math.cos(index * 0.6),
            rim_roll_deg=9.0 * math.sin(index * 1.7),
            pattern_size=PATTERN,
            square_size=SQUARE,
        )
        frames.append(
            render_board(
                rvec,
                tvec,
                true_camera_matrix(),
                TRUE_DIST,
                TRUE_SIZE,
                pattern_size=PATTERN,
                square_size=SQUARE,
            )
        )
        poses.append((rvec, tvec))
    return frames, poses


def _sample_body_poses(count: int, seed: int = 3) -> list[tuple[np.ndarray, np.ndarray]]:
    """生成有足够姿态多样性的机体位姿（棋盘格固定在世界原点）。

    机体位置要让棋盘格中心落在像面中心附近并占据足够像素：``X`` 的杆臂很小，
    所以机体的 ``(north, east)`` 基本就是"相机相对棋盘格的横向偏移"，
    把它设成 0 才让板子居中（板子中心=棋盘原点）。
    """
    rng = np.random.default_rng(seed)
    distance = 0.28
    poses: list[tuple[np.ndarray, np.ndarray]] = []
    for index in range(count):
        roll = 12.0 * math.sin(index * 0.7)
        pitch = 10.0 * math.cos(index * 0.5)
        yaw = (index * 37.0) % 360.0
        rotation = rotation_z(yaw) @ rotation_y(pitch) @ rotation_x(roll)
        # 横向偏移很小（保持板子在画面内），高度在标称距离附近抖动
        position = np.array(
            [
                rng.normal(scale=0.008),
                rng.normal(scale=0.008),
                -distance + rng.normal(scale=0.01),
            ]
        )
        poses.append((rotation, position))
    return poses


# ----------------------------------------------------------------------
# 步骤一：内参
# ----------------------------------------------------------------------
def test_intrinsics_recovered_from_synthetic_views() -> None:
    """合成多视图 → 内参应当能解回真值（相对误差 < 2%）。"""
    frames, _ = _render_views(20)
    result = run_intrinsics(frames, pattern_size=PATTERN, square_size=SQUARE, min_views=5)
    assert result.views >= 5, f"只检出 {result.views} 个视图"
    assert result.rms < 1.0, f"重投影 RMS 过大: {result.rms}"
    assert result.camera_matrix[0, 0] == pytest.approx(TRUE_FX, rel=0.02)
    assert result.camera_matrix[1, 1] == pytest.approx(TRUE_FY, rel=0.02)
    assert result.camera_matrix[0, 2] == pytest.approx(TRUE_CX, abs=12.0)
    assert result.camera_matrix[1, 2] == pytest.approx(TRUE_CY, abs=12.0)


def test_intrinsics_needs_enough_views() -> None:
    blank = [np.full((720, 1280, 3), 128, np.uint8) for _ in range(4)]
    with pytest.raises(ValueError, match="有效视图不足"):
        run_intrinsics(blank, pattern_size=PATTERN, square_size=SQUARE, min_views=5)


def test_intrinsics_uses_more_than_sixty_cached_views() -> None:
    """完整采集视图都进入一次联合内参求解，不再截到旧上限。"""
    from airdrop_calibration.algorithms import run_intrinsics_from_corners

    frames, _ = _render_views(65)
    observations = []
    for frame in frames:
        corners = detect_corners(frame, PATTERN)
        assert corners is not None
        assert corners.dtype == np.float64
        observations.append((TRUE_SIZE, corners))
    result = run_intrinsics_from_corners(
        observations, pattern_size=PATTERN, square_size=SQUARE, min_views=5
    )
    assert result.views == 65
    assert result.camera_matrix[0, 0] == pytest.approx(TRUE_FX, rel=0.02)
    assert result.camera_matrix[1, 1] == pytest.approx(TRUE_FY, rel=0.02)


def test_motion_pair_sampling_preserves_order_without_quadratic_candidates() -> None:
    """抽样仍等价于旧字典序步长，同时大 N 不先构造全部配对。"""
    from airdrop_calibration.algorithms import _motion_pairs as sampled_pairs

    for count in range(16):
        all_pairs = [
            (first, second) for first in range(count) for second in range(first + 1, count)
        ]
        for maximum in (1, 2, 8, 2_000):
            stride = len(all_pairs) // maximum + 1 if len(all_pairs) > maximum else 1
            assert sampled_pairs(count, maximum) == all_pairs[::stride]
    large_pairs = sampled_pairs(100_000, 2_000)
    assert len(large_pairs) <= 2_000
    assert large_pairs[0] == (0, 1)


def test_detect_corners_finds_rendered_board() -> None:
    frames, _ = _render_views(1)
    corners = detect_corners(frames[0], PATTERN)
    assert corners is not None, "渲染的棋盘格应当能被检出"
    assert corners.shape == (PATTERN[0] * PATTERN[1], 1, 2)


def test_load_flight_frames_keeps_each_image_paired_with_its_own_timestamp(
    workdir: Path,
) -> None:
    """帧与拍摄时刻必须来自**同一条**索引记录（坏帧不许造成错位）。

    曾经是两个独立函数：一个按"成功解码数"计数并跳过坏帧，另一个按"索引行数"
    计数。素材中间有 2 帧损坏时两个列表长度仍相等、``zip(strict=True)`` 不报错，
    但 ``frames[k]`` 配的是另一行的时刻——实测造成 0.2s 错位，Δt 与手眼外参
    全建立在错配上，产出一份看起来正常的错误 ``camera_calib.json``。
    """
    from airdrop_calibration.flight_log import iter_flight_frames_paired, load_flight_index

    flight = workdir / "flight"
    (flight / "frames").mkdir(parents=True)
    bad = {2, 3}
    count = 10
    lines = []
    for i in range(count):
        name = f"{i:06d}.jpg"
        if i in bad:
            (flight / "frames" / name).write_bytes(b"not-a-jpeg")  # 存在但解不出
        else:
            cv2.imwrite(
                str(flight / "frames" / name),
                np.full((64, 64, 3), (i * 20) % 255, np.uint8),
            )
        lines.append(
            json.dumps(
                {
                    "filename": name,
                    "capture_timestamp": 100.0 + i * 0.1,
                    # 本用例只验"图像与时刻同源"，两条时间轴取同一个值（lag=0）即可
                    "received_timestamp": 100.0 + i * 0.1,
                    "lag": 0.0,
                }
            )
        )
    (flight / "frames_index.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    truth = {100.0 + i * 0.1: i for i in range(count)}
    paired = list(iter_flight_frames_paired(flight, limit=5))
    assert len(paired) == 5
    for record, filename, image in paired:
        assert image is not None
        # 这一帧的真实索引行必须与它带的时刻一致
        expected_row = truth[record.received_timestamp]
        assert expected_row not in bad, (
            f"时刻 {record.received_timestamp} 属于被损坏的行 {expected_row}"
        )
        # 像素值能反查回它是第几帧 ⇒ 图像与时刻确实同源
        assert int(image[0, 0, 0]) == (expected_row * 20) % 255
        # 两条时间轴与当时的 lag 跟着记录一起回来（标定要挑轴，见 load_flight_stamps）
        assert record.lag_s == 0.0
        assert record.capture_timestamp == pytest.approx(record.received_timestamp)

    # 索引里解不出的两行被连同其时刻一起跳过，而不是留下空洞
    assert [round(record.received_timestamp, 3) for record, _, _ in paired] == [
        100.0,
        100.1,
        100.4,
        100.5,
        100.6,
    ]
    assert len(load_flight_index(flight)) == count


def test_flight_index_default_reads_beyond_200_and_limit_counts_records(
    workdir: Path,
) -> None:
    """帧时间辅助接口默认读完整索引，显式限制只计非空记录。"""
    from airdrop_calibration.flight_log import load_flight_timestamps

    flight = workdir / "long-index"
    flight.mkdir()
    rows = [
        json.dumps(
            {
                "filename": f"{index:06d}.jpg",
                "capture_timestamp": float(index),
                "received_timestamp": float(index) + 0.1,
            }
        )
        for index in range(205)
    ]
    (flight / "frames_index.jsonl").write_text("\n" + "\n\n".join(rows) + "\n", encoding="utf-8")
    timestamps = load_flight_timestamps(flight)
    assert len(timestamps) == 205
    assert timestamps[-1] == pytest.approx(204.1)
    assert load_flight_timestamps(flight, limit=2) == pytest.approx([0.1, 1.1])


def test_flight_index_time_bases_and_missing_received(workdir: Path) -> None:
    """帧索引的两条时间轴：默认给 received（标定用），capture 只做诊断；缺字段显式报错。"""
    from airdrop_calibration.flight_log import load_flight_index, load_flight_stamps

    flight = workdir / "flight"
    flight.mkdir()
    (flight / "frames_index.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "filename": f"{i:06d}.jpg",
                    "capture_timestamp": 100.0 + i,
                    "received_timestamp": 100.5 + i,
                    "lag": 0.5,
                }
            )
            for i in range(3)
        )
        + "\n",
        encoding="utf-8",
    )
    stamps = load_flight_stamps(flight)
    assert [item.received_timestamp for item in stamps] == [100.5, 101.5, 102.5]
    assert [item.capture_timestamp for item in stamps] == [100.0, 101.0, 102.0]
    assert all(item.lag_s == 0.5 for item in stamps)
    # 默认（received）与显式 capture 各给一条轴
    assert [stamp for stamp, _ in load_flight_index(flight)] == [100.5, 101.5, 102.5]
    assert [stamp for stamp, _ in load_flight_index(flight, time_base="capture")] == [
        100.0,
        101.0,
        102.0,
    ]
    with pytest.raises(ValueError, match="time_base"):
        load_flight_index(flight, time_base="camera")

    # 老素材（缺 received_timestamp）不许静默退回 capture 轴
    (flight / "frames_index.jsonl").write_text(
        json.dumps({"filename": "000000.jpg", "capture_timestamp": 100.0}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="received_timestamp"):
        load_flight_stamps(flight)


def _snapshots_from_rotations(
    times: Sequence[float], rotations: Sequence[np.ndarray]
) -> list[TelemetrySnapshot]:
    """按给定时刻与姿态造一批遥测快照（只需 timestamp + 四元数）。"""
    out = []
    for stamp, rotation in zip(times, rotations, strict=True):
        quaternion = Rotation.from_matrix(rotation).as_quat()  # scipy 是 (x, y, z, w)
        out.append(
            TelemetrySnapshot(
                timestamp=float(stamp),
                quaternion_w=float(quaternion[3]),
                quaternion_x=float(quaternion[0]),
                quaternion_y=float(quaternion[1]),
                quaternion_z=float(quaternion[2]),
            )
        )
    return out


def test_lag_measurement_is_absolute_on_received_axis() -> None:
    """时间差必须在 **received 轴**上测：结果 = ``d_video − d_tel``，与录制时的 lag 无关。

    为什么（真机上会踩的坑）：``capture_timestamp`` 是录制时按配置的 ``telemetry_lag``
    **假设**出来的拍摄时刻（= received − lag）。拿它去测，结果里必然带着"假设的残差"——
    换一次配置就平移一个 lag，表现为"把上次标定值回填再录一次，Δt 自己收敛到 0"，
    但那不代表标定对了。received 是原始量，测出来的才是可直接回填运行时的绝对对齐量。

    本用例直接喂受控序列（不渲染）：同一批姿态，只改录制时的 lag，看两条轴分别怎么变。
    """

    count = 30
    frame_dt = 1.0 / 30.0
    t0 = 1_000_000.0
    physical = [t0 + index * frame_dt for index in range(count)]
    rotations = [
        rotation_z(25.0 * math.sin(2.0 * math.pi * index / count))
        @ rotation_x(15.0 * math.cos(3.0 * math.pi * index / count))
        for index in range(count)
    ]
    d_video, d_tel = 0.15, 0.04
    snapshots = _snapshots_from_rotations([stamp + d_tel for stamp in physical], rotations)

    def measure(lag_cfg: float):
        received = [stamp + d_video for stamp in physical]
        capture = [stamp - lag_cfg for stamp in received]
        return measure_frame_telemetry_lag(
            received, capture, rotations, snapshots, baseline_s=lag_cfg
        )

    first = measure(0.15)
    assert first.absolute.lag_s == pytest.approx(d_video - d_tel, abs=0.01)
    assert first.residual is not None
    assert first.residual.lag_s == pytest.approx((d_video - 0.15) - d_tel, abs=0.01)
    assert first.cross_check_ok  # 基准 + 残差 = 绝对量
    assert first.time_base == "received"

    # 只改录制时的 lag：capture 轴平移一个 lag，received 轴**不动**
    second = measure(0.05)
    assert second.absolute.lag_s == pytest.approx(first.absolute.lag_s, abs=0.005)
    assert second.residual is not None
    assert second.residual.lag_s == pytest.approx(first.residual.lag_s + 0.10, abs=0.01)
    assert second.cross_check_ok


def test_rendered_board_projects_inside_frame() -> None:
    """板子四角必须投影在画幅内——被裁掉一半的板子检不出角点。"""
    frames, poses = _render_views(8)
    board = board_object_points(PATTERN, SQUARE)
    for rvec, tvec in poses:
        projected, _ = cv2.projectPoints(board, rvec, tvec, true_camera_matrix(), TRUE_DIST)
        points = projected.reshape(-1, 2)
        assert points[:, 0].min() > 0 and points[:, 0].max() < TRUE_SIZE[0]
        assert points[:, 1].min() > 0 and points[:, 1].max() < TRUE_SIZE[1]


def test_look_at_board_centers_the_board() -> None:
    """板子几何中心应当投影到主点附近（对准了才不会被裁）。"""
    cols, rows = PATTERN
    center = np.array([[[(cols - 1) * SQUARE / 2, (rows - 1) * SQUARE / 2, 0.0]]], dtype=np.float64)
    rvec, tvec = look_at_board(0.55, true_camera_matrix(), pattern_size=PATTERN, square_size=SQUARE)
    projected, _ = cv2.projectPoints(center, rvec, tvec, true_camera_matrix(), np.zeros(5))
    assert float(projected[0, 0, 0]) == pytest.approx(TRUE_CX, abs=1.0)
    assert float(projected[0, 0, 1]) == pytest.approx(TRUE_CY, abs=1.0)


def test_detect_corners_returns_none_on_blank() -> None:
    assert detect_corners(np.full((480, 640, 3), 127, np.uint8), PATTERN) is None


# ----------------------------------------------------------------------
# 步骤二：时间差
# ----------------------------------------------------------------------
def test_lag_recovered_from_shifted_series() -> None:
    """把机体角速度序列整体平移已知量，互相关应当把它解回来。"""
    step = 0.01
    times = np.arange(0.0, 3.0, step)
    # 一段有明显起伏的角速度（避免常数序列相关为 0）
    body_speed = (
        20.0 + 15.0 * np.sin(2 * math.pi * 1.5 * times) + 5.0 * np.sin(2 * math.pi * 5.0 * times)
    )
    true_lag = 0.12
    camera_times = times + true_lag  # 画面比姿态晚 true_lag
    camera_speed = body_speed.copy()
    result = estimate_lag_by_correlation(
        camera_times,
        camera_speed,
        times,
        body_speed,
        search_s=0.5,
        step_s=0.005,
    )
    assert result.ok
    assert result.lag_s == pytest.approx(true_lag, abs=0.02), f"解出 {result.lag_s}"
    assert result.peak > 0.9


@pytest.mark.parametrize("true_lag", [0.0, -0.15, 0.25])
def test_lag_recovers_various_offsets(true_lag: float) -> None:
    step = 0.01
    times = np.arange(0.0, 3.0, step)
    body_speed = 25.0 + 18.0 * np.sin(2 * math.pi * 1.2 * times)
    camera_times = times + true_lag
    result = estimate_lag_by_correlation(
        camera_times,
        body_speed.copy(),
        times,
        body_speed,
        search_s=0.5,
        step_s=0.005,
    )
    assert result.lag_s == pytest.approx(true_lag, abs=0.02)


def test_lag_reports_failure_on_flat_series() -> None:
    times = np.arange(0.0, 2.0, 0.01)
    flat = np.full_like(times, 30.0)
    result = estimate_lag_by_correlation(times, flat, times, flat)
    assert not result.ok


def test_lag_reports_failure_on_too_few_samples() -> None:
    result = estimate_lag_by_correlation(
        np.array([0.0, 0.1]),
        np.array([1.0, 2.0]),
        np.array([0.0, 0.1]),
        np.array([1.0, 2.0]),
    )
    assert not result.ok


def test_angular_speed_from_poses_matches_constant_rate() -> None:
    """匀速旋转 30°/s：算出的角速度应当处处是 30°/s。"""
    rate = 30.0
    step = 0.02
    times = np.arange(0.0, 1.0, step)
    rotations = [rotation_z(rate * t) for t in times]
    out_times, speeds = angular_speed_from_poses(times, rotations)  # pyright: ignore[reportArgumentType]
    assert out_times.size == times.size - 1
    assert np.allclose(speeds, rate, atol=1e-6)


def test_rotation_angle_and_log_round_trip() -> None:
    for axis in (
        np.array([1.0, 0, 0]),
        np.array([0, 1.0, 0]),
        np.array([0, 0, 1.0]),
        np.array([1.0, 2.0, -0.5]) / np.linalg.norm([1.0, 2.0, -0.5]),
    ):
        for angle_deg in (0.1, 5.0, 45.0, 179.0):
            vector = axis * math.radians(angle_deg)
            matrix = _axis_angle_to_matrix(vector)
            assert rotation_angle_deg(matrix) == pytest.approx(angle_deg, abs=1e-6)
            assert np.allclose(rotation_log(matrix), vector, atol=1e-6)


def _axis_angle_to_matrix(vector: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(vector))
    if theta < 1e-12:
        return np.eye(3)
    axis = vector / theta
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(theta) * k + (1 - math.cos(theta)) * (k @ k)


# ----------------------------------------------------------------------
# 步骤三：手眼（纯算法）
# ----------------------------------------------------------------------
def true_x_matrix() -> np.ndarray:
    """真值 ``X = T_body_cam = [R_bc | t_bc]``（相机→机体，与 georef 同语义）。"""
    matrix = np.eye(4)
    matrix[:3, :3] = true_r_bc()
    matrix[:3, 3] = TRUE_T_BC
    return matrix


def pose_of(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))[0]
    matrix[:3, 3] = np.asarray(tvec, float).reshape(3)
    return matrix


def body_pose_from_board(board_pose: np.ndarray, x_true: np.ndarray) -> np.ndarray:
    """由"棋盘格在相机系"的位姿导出"机体在世界系"的位姿。

    棋盘格固定在世界原点（``C = T_world_board = I``），于是
    ``G X P = C`` ⇒ ``G = P⁻¹ X⁻¹``。

    ⚠ 这里只有一个正确方向。写成 ``G = X·P``（等价于把"棋盘格在机体系"当成机体位姿）
    就等于是拿未知量当已知量用：相对运动里那个常量会被整体消掉，
    方程退化成"求与所有 R 可交换的矩阵"，真值根本不在解空间里。
    回归用例 ``test_body_pose_must_come_from_telemetry`` 把它钉死。
    """
    return np.linalg.inv(board_pose) @ np.linalg.inv(x_true)


def _motion_pairs(
    bodies: Sequence[np.ndarray], cameras: Sequence[np.ndarray]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """``A_ij = G_j⁻¹G_i``（遥测）、``B_ij = P_jP_i⁻¹``（PnP）——与 run_extrinsics 一致。"""
    a_list: list[np.ndarray] = []
    b_list: list[np.ndarray] = []
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            a_list.append(np.linalg.inv(bodies[j]) @ bodies[i])
            b_list.append(cameras[j] @ np.linalg.inv(cameras[i]))
    return a_list, b_list


def _board_views(
    count: int = 24,
    distance: float = 0.55,
    *,
    pattern_size: tuple[int, int] = PATTERN,
):
    """渲染一批"相机对着板子"的视图，返回 (帧, 棋盘格→相机 位姿)。"""
    frames: list[np.ndarray] = []
    poses: list[np.ndarray] = []
    for index in range(count):
        rvec, tvec = look_at_board(
            distance,
            true_camera_matrix(),
            tilt_deg=16.0 * math.sin(index * 0.9),
            yaw_deg=13.0 * math.cos(index * 0.6),
            rim_roll_deg=10.0 * math.sin(index * 1.7),
            pattern_size=pattern_size,
            square_size=SQUARE,
        )
        frames.append(
            render_board(
                rvec,
                tvec,
                true_camera_matrix(),
                TRUE_DIST,
                TRUE_SIZE,
                pattern_size=pattern_size,
                square_size=SQUARE,
            )
        )
        poses.append(pose_of(rvec, tvec))
    return frames, poses


def _random_rotations(count: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    rotations = []
    for _ in range(count):
        vector = rng.normal(size=3)
        vector = vector / np.linalg.norm(vector) * math.radians(rng.uniform(-60, 60))
        rotations.append(_axis_angle_to_matrix(vector))
    return rotations


# 多组种子：解法的"零空间符号"曾出过 bug（真值在零空间里，却解出精确 180° 的那个版本），
# 单组数据有 50% 概率蒙对，所以这里跑多组。
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_solver_recovers_known_transform(seed: int) -> None:
    """``A`` 来遥测、``B`` 来 PnP 时，``AX=XB`` 应当解到机器精度。"""
    x_true = true_x_matrix()
    rng = np.random.default_rng(seed)
    bodies = []
    for rotation in _random_rotations(12, seed):
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = rng.normal(scale=0.3, size=3)  # 机体位置是独立量
        bodies.append(matrix)
    cameras = [np.linalg.inv(x_true) @ np.linalg.inv(g) for g in bodies]
    a_list, b_list = _motion_pairs(bodies, cameras)

    rotation, translation, smallest, spread = solve_hand_eye(a_list, b_list)
    assert smallest < 1e-9, f"激励充足时最小奇异值应接近 0，实际 {smallest}"
    assert spread > 15.0
    assert translation is not None
    assert np.allclose(rotation, true_r_bc(), atol=1e-9), np.abs(rotation - true_r_bc()).max()
    assert np.allclose(translation, TRUE_T_BC, atol=1e-9), np.abs(translation - TRUE_T_BC).max()


def test_body_pose_must_come_from_telemetry() -> None:
    """反例固化：机体位姿若由棋盘格位姿左乘常量导出，就解不出真值。

    这不是"板子没转"（板子固定、飞机在动时姿态当然在变），而是把未知量当已知量用。
    """
    x_true = true_x_matrix()
    rng = np.random.default_rng(7)
    cameras = []
    for index, rotation in enumerate(_random_rotations(10, 7)):
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = rng.normal(scale=0.3, size=3)
        cameras.append(matrix)
    bodies = [x_true @ p for p in cameras]  # 错误方向：G = X·P
    a_list, b_list = _motion_pairs(bodies, cameras)
    rotation, *_ = solve_hand_eye(a_list, b_list)
    error = rotation_angle_deg(rotation.T @ true_r_bc())
    assert error > 10.0, f"错误构造居然解出了真值（误差 {error:.3f}°），用例失效"


def test_solver_requires_pairs() -> None:
    with pytest.raises(ValueError, match="运动对数量不足"):
        solve_hand_eye([np.eye(4)], [np.eye(4)])


def test_extrinsics_rejects_single_rotation_axis() -> None:
    """所有运动对绕同一根轴时解不唯一（OpenCV 文档同样要求 ≥2 个不平行轴）。"""
    x_true = true_x_matrix()
    bodies = []
    for index in range(12):
        matrix = np.eye(4)
        matrix[:3, :3] = rotation_z(7.0 * index)  # 只绕 z 转
        matrix[:3, 3] = np.array([0.01 * index, 0.0, -0.45])
        bodies.append(matrix)
    cameras = [np.linalg.inv(x_true) @ np.linalg.inv(g) for g in bodies]
    board_poses = [
        (float(index), cv2.Rodrigues(p[:3, :3])[0], p[:3, 3]) for index, p in enumerate(cameras)
    ]
    with pytest.raises(ValueError, match="旋转轴过于单一"):
        run_extrinsics(board_poses=board_poses, body_poses=[pose[:3, :3] for pose in bodies])


def test_extrinsics_requires_one_body_pose_per_board_pose() -> None:
    """遥测 10 Hz、画面 30 Hz，不能"第 i 条遥测配第 i 帧"——数量对不上必须报错。"""
    x_true = true_x_matrix()
    frames, poses = _board_views(6)
    bodies = [body_pose_from_board(p, x_true) for p in poses]
    board_poses = [
        (float(index), cv2.Rodrigues(p[:3, :3])[0], p[:3, 3]) for index, p in enumerate(poses)
    ]
    with pytest.raises(ValueError, match="数量不一致"):
        run_extrinsics(board_poses=board_poses, body_poses=[pose[:3, :3] for pose in bodies[:3]])


# ----------------------------------------------------------------------
# 步骤三：遥测位姿的重采样与配对
# ----------------------------------------------------------------------
def _yaw_snapshot(stamp: float, yaw_deg: float, down: float = -0.45) -> TelemetrySnapshot:
    half = math.radians(yaw_deg) / 2.0
    return TelemetrySnapshot(
        timestamp=stamp,
        north_m=0.0,
        east_m=0.0,
        down_m=down,
        quaternion_w=math.cos(half),
        quaternion_x=0.0,
        quaternion_y=0.0,
        quaternion_z=math.sin(half),
    )


def test_telemetry_pose_at_interpolates_and_refuses_extrapolation() -> None:
    snapshots = [_yaw_snapshot(index * 0.1, 10.0 * index) for index in range(5)]
    pose = telemetry_pose_at(snapshots, 0.05)
    assert pose is not None
    assert rotation_angle_deg(pose[:3, :3].T @ rotation_z(5.0)) < 1e-9
    assert telemetry_pose_at(snapshots, -0.1) is None  # 不外推
    assert telemetry_pose_at(snapshots, 10.0) is None


def test_align_body_poses_applies_lag_with_the_right_sign() -> None:
    """``lag > 0`` 表示画面滞后 ⇒ 该帧查的是更早的遥测。

    造一段匀速偏航的遥测，让棋盘格位姿对应 ``capture_timestamp − lag`` 时刻的机体姿态；
    带上 lag 配对应当精确复原，不带就有 ``lag·ω`` 的固定偏差。
    """
    rate, step, lag = 40.0, 0.02, 0.10  # °/s, s, s
    snapshots = [_yaw_snapshot(index * step, rate * index * step) for index in range(60)]
    x_true = true_x_matrix()
    board_poses = []
    for index in range(5, 50):
        stamp = index * step
        body = np.eye(4)
        body[:3, :3] = rotation_z(rate * (stamp - lag))  # 真实拍摄时刻的机体姿态
        body[:3, 3] = np.array([0.0, 0.0, -0.45])
        camera = np.linalg.inv(x_true) @ np.linalg.inv(body)
        board_poses.append((stamp, cv2.Rodrigues(camera[:3, :3])[0], camera[:3, 3]))

    paired_board, paired_body = align_body_poses(board_poses, snapshots, lag_s=lag)
    assert len(paired_board) == len(paired_body) > 20
    expected = rotation_z(rate * (paired_board[0][0] - lag))
    error = rotation_angle_deg(paired_body[0][:3, :3].T @ expected)
    assert error < 0.05, f"扣掉 lag 后应当复原，实际差 {error:.3f}°"

    _, wrong_body = align_body_poses(board_poses, snapshots, lag_s=0.0)
    drift = rotation_angle_deg(wrong_body[0][:3, :3].T @ expected)
    assert drift > 3.0, f"不扣 lag 应当有 {rate * lag:.1f}° 的偏差，实际 {drift:.3f}°"


# ----------------------------------------------------------------------
# 渲染链路的自洽性（棋盘格相位）
# ----------------------------------------------------------------------
def test_rendered_board_matches_requested_pose() -> None:
    """渲染出来的棋盘格必须与给定位姿同相位。

    回归用例：to_board 少减一个 tile 时，棋盘格相对对象点整体错开一整格，
    检测出的角点落在"板子真实点 +1 格"处——重投影误差仍然只有 0.1 px，
    但位姿差 180°，手眼标定直接报废，靠重投影根本发现不了。只有方格相位能发现
    （实测错格时相符率 < 0.5，正确时 1.000）。
    """
    pattern_size = ORIENTABLE_PATTERN
    board = board_object_points(pattern_size, SQUARE)
    for index in (0, 3, 7, 11):
        rvec, tvec = look_at_board(
            0.55,
            true_camera_matrix(),
            tilt_deg=16.0 * math.sin(index * 0.9),
            yaw_deg=13.0 * math.cos(index * 0.6),
            rim_roll_deg=10.0 * math.sin(index * 1.7),
            pattern_size=pattern_size,
            square_size=SQUARE,
        )
        frame = render_board(
            rvec,
            tvec,
            true_camera_matrix(),
            TRUE_DIST,
            TRUE_SIZE,
            pattern_size=pattern_size,
            square_size=SQUARE,
        )
        ratio = pattern_phase_ok(
            frame,
            rvec,
            tvec,
            true_camera_matrix(),
            TRUE_DIST,
            pattern_size=pattern_size,
            square_size=SQUARE,
        )
        assert ratio > 0.9, f"第 {index} 帧相位相符率只有 {ratio:.2f}：棋盘格与对象点错格了"

        pose = solve_board_poses(
            [frame],
            board,
            true_camera_matrix(),
            TRUE_DIST,
            pattern_size=pattern_size,
            square_size=SQUARE,
        )[0]
        assert pose is not None, f"第 {index} 帧应当能解出位姿"
        corners = detect_corners(frame, pattern_size)
        assert corners is not None
        projected, _ = cv2.projectPoints(board, pose[0], pose[1], true_camera_matrix(), TRUE_DIST)
        # cv2 的类型桩把 projectPoints 的结果标成可能为 None，运行时它总是 ndarray；
        # 这里显式收窄，顺带让后面那行减法/平方有明确类型。
        assert projected is not None
        residual = (projected - corners) ** 2
        rms = float(np.sqrt(np.mean(np.sum(residual, axis=2))))
        assert rms < 0.5, f"第 {index} 帧解出的位姿重投影残差 {rms:.3f} px"


def test_pattern_phase_selects_direction_for_opposite_parity_board() -> None:
    """异奇偶内角点棋盘的相位应区分正序和反序候选。"""
    frames, camera_poses = _board_views(8, pattern_size=ORIENTABLE_PATTERN)
    frame = frames[3]
    rvec = cv2.Rodrigues(camera_poses[3][:3, :3])[0]
    tvec = camera_poses[3][:3, 3]
    board = board_object_points(ORIENTABLE_PATTERN, SQUARE)
    corners = detect_corners(frame, ORIENTABLE_PATTERN)
    assert corners is not None
    good = pattern_phase_ok(
        frame,
        rvec,
        tvec,
        true_camera_matrix(),
        TRUE_DIST,
        pattern_size=ORIENTABLE_PATTERN,
        square_size=SQUARE,
    )
    # 另一个标签支：把检测到的角点顺序整体反过来的那个解
    ok, other_rvec, other_tvec = cv2.solvePnP(
        board,
        corners[::-1],
        true_camera_matrix(),
        TRUE_DIST,
        flags=cv2.SOLVEPNP_IPPE,
    )
    assert ok
    assert other_rvec is not None and other_tvec is not None
    assert (
        rotation_angle_deg(pose_of(other_rvec, other_tvec)[:3, :3].T @ pose_of(rvec, tvec)[:3, :3])
        > 170.0
    ), "反序解应当与正序解差 180°"
    flipped = pattern_phase_ok(
        frame,
        other_rvec,
        other_tvec,
        true_camera_matrix(),
        TRUE_DIST,
        pattern_size=ORIENTABLE_PATTERN,
        square_size=SQUARE,
    )
    assert good > 0.9, f"正确位姿的相位相符率只有 {good:.2f}"
    assert flipped < 0.1, f"反序候选不应通过相位门槛，实际 {flipped:.2f}"
    selected = solve_board_poses(
        [frame],
        board,
        true_camera_matrix(),
        TRUE_DIST,
        pattern_size=ORIENTABLE_PATTERN,
        square_size=SQUARE,
    )[0]
    assert selected is not None
    assert rotation_angle_deg(cv2.Rodrigues(selected[0])[0].T @ cv2.Rodrigues(rvec)[0]) < 2.0


@pytest.mark.parametrize("turn_deg", [120.0, 180.0])
def test_pattern_phase_tracks_large_real_turn_without_continuity_guess(turn_deg: float) -> None:
    """连续帧转过大角度时仍按各帧棋盘相位独立判向。"""
    pattern = ORIENTABLE_PATTERN
    base_rvec, base_tvec = look_at_board(
        0.55, true_camera_matrix(), pattern_size=pattern, square_size=SQUARE
    )
    base_rotation = cv2.Rodrigues(base_rvec)[0]
    center = np.array([(pattern[0] * SQUARE) / 2, (pattern[1] * SQUARE) / 2, 0.0])
    images: list[np.ndarray] = []
    expected: list[np.ndarray] = []
    for angle in (0.0, turn_deg):
        rotation = base_rotation @ rotation_z(angle)
        tvec = (base_tvec.reshape(3) + base_rotation @ center - rotation @ center).reshape(3, 1)
        rvec = cv2.Rodrigues(rotation)[0]
        images.append(
            render_board(
                rvec,
                tvec,
                true_camera_matrix(),
                TRUE_DIST,
                TRUE_SIZE,
                pattern_size=pattern,
                square_size=SQUARE,
            )
        )
        expected.append(rotation)
    poses = solve_board_poses(
        images,
        board_object_points(pattern, SQUARE),
        true_camera_matrix(),
        TRUE_DIST,
        pattern_size=pattern,
        square_size=SQUARE,
    )
    assert all(pose is not None for pose in poses)
    for pose, truth in zip(poses, expected, strict=True):
        assert pose is not None
        solved_rotation = cv2.Rodrigues(pose[0])[0]
        assert rotation_angle_deg(solved_rotation.T @ truth) < 4.0


@pytest.mark.parametrize("pattern", [(9, 6), (8, 5)])
def test_solve_board_poses_accepts_opposite_parity_dimensions(pattern: tuple[int, int]) -> None:
    """两种异奇偶方向都应返回通过黑白相位判定的位姿。"""
    frame = render_board(
        np.array([[0.2], [-0.3], [0.1]]),
        np.array([[0.0], [0.0], [0.7]]),
        true_camera_matrix(),
        TRUE_DIST,
        TRUE_SIZE,
        pattern_size=pattern,
        square_size=SQUARE,
    )
    points = board_object_points(pattern, SQUARE)
    pose = solve_board_poses(
        [frame],
        points,
        true_camera_matrix(),
        TRUE_DIST,
        pattern_size=pattern,
        square_size=SQUARE,
    )[0]
    assert pose is not None


def test_solve_board_poses_rejects_same_parity_and_low_contrast() -> None:
    """不可辨方向的奇偶尺寸报错，低对比度图像不产生伪位姿。"""
    with pytest.raises(ValueError, match="必须一奇一偶"):
        solve_board_poses(
            [],
            board_object_points((8, 6), SQUARE),
            true_camera_matrix(),
            TRUE_DIST,
            pattern_size=(8, 6),
        )
    frame = render_board(
        np.array([[0.2], [-0.3], [0.1]]),
        np.array([[0.0], [0.0], [0.7]]),
        true_camera_matrix(),
        TRUE_DIST,
        TRUE_SIZE,
        pattern_size=ORIENTABLE_PATTERN,
        square_size=SQUARE,
    )
    low_contrast = np.clip(128 + (frame.astype(np.float32) - 128) * 0.1, 0, 255).astype(np.uint8)
    assert solve_board_poses(
        [low_contrast],
        board_object_points(ORIENTABLE_PATTERN, SQUARE),
        true_camera_matrix(),
        TRUE_DIST,
        pattern_size=ORIENTABLE_PATTERN,
        square_size=SQUARE,
    ) == [None]


# ----------------------------------------------------------------------
# 三步串起来：渲染 → 检测（含相位消解）→ 配对 → 解外参
# ----------------------------------------------------------------------
def test_full_pipeline_recovers_extrinsics_from_synthetic_views() -> None:
    """端到端：像素数据里含有真值外参，整条链路要能把它解回来。

    链路：渲染棋盘格 → 检测 + 相位消解标签 → PnP 得 ``P`` → 由 ``G = P⁻¹X⁻¹`` 得
    机体位姿（模拟"板子固定、飞机在动"的采集）→ ``AX=XB`` 解 ``X``。
    """
    x_true = true_x_matrix()
    board = board_object_points(ORIENTABLE_PATTERN, SQUARE)
    frames, camera_poses = _board_views(24, pattern_size=ORIENTABLE_PATTERN)

    intrinsics = run_intrinsics(
        frames, pattern_size=ORIENTABLE_PATTERN, square_size=SQUARE, min_views=8
    )
    assert intrinsics.rms < 1.0
    assert intrinsics.camera_matrix[0, 0] == pytest.approx(TRUE_FX, rel=0.02)

    solved_raw = solve_board_poses(
        frames,
        board,
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
        pattern_size=ORIENTABLE_PATTERN,
        square_size=SQUARE,
    )
    solved: list[tuple[float, np.ndarray, np.ndarray]] = []
    branches: set[bool] = set()
    for index, pose in enumerate(solved_raw):
        assert pose is not None, f"第 {index} 帧应当能解出位姿"
        rotation = cv2.Rodrigues(pose[0])[0]
        truth = camera_poses[index][:3, :3]
        forward = rotation_angle_deg(rotation.T @ truth)
        mirrored = rotation_angle_deg(rotation.T @ (truth @ rotation_z(180.0)))
        assert min(forward, mirrored) < 1.0, (
            f"第 {index} 帧既不是正序也不是反序解: {forward:.3f}° / {mirrored:.3f}°"
        )
        branches.add(forward < mirrored)
        solved.append((float(index), pose[0], pose[1]))
    assert len(branches) == 1, "标签支必须全程一致——混用会让 AX=XB 崩掉"

    bodies = [body_pose_from_board(p, x_true) for p in camera_poses]
    extrinsics = run_extrinsics(board_poses=solved, body_poses=[pose[:3, :3] for pose in bodies])
    assert extrinsics.ok
    error = rotation_angle_deg(extrinsics.r_bc.T @ true_r_bc())
    assert error < 3.0, f"外参旋转误差 {error:.3f}° 过大"
    assert extrinsics.axis_spread_deg > 15.0
    assert extrinsics.pairs >= 10


def test_full_pipeline_rejects_insufficient_rotation_excitation() -> None:
    """旋转激励不足时必须明确报错，而不是给一个看起来"成功"的解。"""
    x_true = true_x_matrix()
    frames, camera_poses = _board_views(8)
    bodies = []
    for index in range(8):
        # 姿态几乎不变（< MIN_PAIR_ROTATION_DEG），只有位置在动
        matrix = np.eye(4)
        matrix[:3, :3] = rotation_z(0.4 * index)
        matrix[:3, 3] = np.array([0.01 * index, 0.0, -0.45])
        bodies.append(matrix)
    camera_poses = [np.linalg.inv(x_true) @ np.linalg.inv(g) for g in bodies]
    board_poses = [
        (float(index), cv2.Rodrigues(p[:3, :3])[0], p[:3, 3])
        for index, p in enumerate(camera_poses)
    ]
    with pytest.raises(ValueError, match="旋转激励不够"):
        run_extrinsics(board_poses=board_poses, body_poses=[pose[:3, :3] for pose in bodies])


# ----------------------------------------------------------------------
# 标定 JSON 的输出契约（杆臂：尺量为准 + 估计值比对）
# ----------------------------------------------------------------------


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    """工作区内的临时目录；用例结束整棵删掉。

    不用 ``tmp_path``：部分受限执行环境里 pytest 的 basetemp 目录建好后不可枚举、
    不可删除（``PermissionError: [WinError 5]``）。
    """
    return tmp_path


def _write_calibration_flight(
    flight_dir: Path,
    count: int = 30,
    *,
    video_delay_s: float = 0.15,
    lag_cfg_s: float = 0.15,
    telemetry_delay_s: float = 0.0,
    pattern_size: tuple[int, int] = PATTERN,
    position_available: bool = True,
    blank_prefix: int = 0,
) -> None:
    """手写一个可标定的飞行目录（渲染棋盘格帧 + 与之自洽的遥测）。

    相机位姿由 :func:`look_at_board` 给出（保证板子全程在视野内），机体位姿取
    ``G = P⁻¹X⁻¹``（棋盘格固定在世界原点，``C = I``）——与合成用例同一套约定，
    遥测里因此天然含真值外参，解不回来就是流水线错了。

    三个延时按参数注入（默认值 = 原先写死的那一组，行为不变）：

    * ``video_delay_s``：真实视频链路延时（``received_timestamp = 物理时刻 + 它``）；
    * ``lag_cfg_s``：录制时 ``VideoConfig.telemetry_lag``（``capture_timestamp = received − 它``）；
    * ``telemetry_delay_s``：遥测链路延时（遥测时间戳 = 物理时刻 + 它）。

    于是标定第 2 步在 **received 轴**上应当测出 ``video_delay_s − telemetry_delay_s``
    （绝对对齐量），而 capture 轴测出的是减去 ``lag_cfg_s`` 的残差。

    遥测在首末各多写一条把帧区间包住：``telemetry_pose_at`` 只做范围内内插、
    拒绝外推，不包住的话首末帧会整帧被丢。
    """
    (flight_dir / "frames").mkdir(parents=True)
    t0 = 1_000_000.0
    frame_dt = 1.0 / 30.0
    lag = lag_cfg_s
    x_true = true_x_matrix()
    index_lines: list[str] = []
    entries: list[tuple[float, np.ndarray]] = []

    for index in range(count):
        pose_index = max(0, index - blank_prefix)
        rvec, tvec = look_at_board(
            0.55,
            true_camera_matrix(),
            tilt_deg=15.0 * math.sin(pose_index * 0.9),
            yaw_deg=12.0 * math.cos(pose_index * 0.6),
            rim_roll_deg=10.0 * math.sin(pose_index * 1.7),
            pattern_size=pattern_size,
            square_size=SQUARE,
        )
        image = render_board(
            rvec,
            tvec,
            true_camera_matrix(),
            TRUE_DIST,
            TRUE_SIZE,
            pattern_size=pattern_size,
            square_size=SQUARE,
        )
        if index < blank_prefix:
            image = np.zeros_like(image)
        name = f"{index:06d}.jpg"
        ok, buffer = cv2.imencode(".jpg", image)
        assert ok
        payload = buffer.tobytes()
        (flight_dir / "frames" / name).write_bytes(payload)
        capture = t0 + index * frame_dt
        index_lines.append(
            json.dumps(
                {
                    "index": index,
                    "filename": name,
                    "capture_timestamp": capture + video_delay_s - lag_cfg_s,
                    "received_timestamp": capture + video_delay_s,  # 与 VideoFrame.timestamp 同义
                    "lag": lag,
                    "extrapolated": False,
                    "offset": 0.0,
                    "bytes": len(payload),
                }
            )
        )
        # 遥测时间戳同样后移一个遥测链路延时（内容仍是该物理时刻的位姿）
        entries.append(
            (
                capture + telemetry_delay_s,
                body_pose_from_board(pose_of(rvec, tvec), x_true),
            )
        )

    (flight_dir / "frames_index.jsonl").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    timeline = [(entries[0][0] - 0.3, entries[0][1])]
    timeline += entries
    timeline.append((entries[-1][0] + 0.3, entries[-1][1]))

    lines = []
    for stamp, pose in timeline:
        quaternion = Rotation.from_matrix(pose[:3, :3]).as_quat()  # scipy 是 (x,y,z,w)
        w, x, y, z = (
            float(quaternion[3]),
            float(quaternion[0]),
            float(quaternion[1]),
            float(quaternion[2]),
        )
        # 钉住约定：写进遥测的 (w,x,y,z) 反解回来必须就是这个旋转矩阵
        assert np.allclose(quaternion_to_matrix_local((w, x, y, z)), pose[:3, :3], atol=1e-9)
        lines.append(
            json.dumps(
                asdict(
                    TelemetrySnapshot(
                        timestamp=stamp,
                        quaternion_received_at=stamp,
                        quaternion_timestamp_us=round((stamp - t0 - telemetry_delay_s + 1) * 1e6),
                        position_velocity_received_at=(stamp if position_available else None),
                        position_timestamp_us=(
                            round((stamp - t0 - telemetry_delay_s + 1) * 1e6)
                            if position_available
                            else None
                        ),
                        north_m=float(pose[0, 3]) if position_available else None,
                        east_m=float(pose[1, 3]) if position_available else None,
                        down_m=float(pose[2, 3]) if position_available else None,
                        quaternion_w=w,
                        quaternion_x=x,
                        quaternion_y=y,
                        quaternion_z=z,
                    )
                )
            )
        )
    (flight_dir / "telemetry.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_calibration_default_covers_frames_after_old_200_frame_prefix(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """完整性、检测和位姿三阶段图像驻留峰值保持常数级。"""
    import weakref

    flight_dir = workdir / "late-boards"
    count = 225
    _write_calibration_flight(
        flight_dir, count=count, pattern_size=ORIENTABLE_PATTERN, blank_prefix=201
    )
    telemetry_count = count + 2
    (flight_dir / "config_snapshot.json").write_text(
        json.dumps({"record": {"telemetry_hz": 0}}), encoding="utf-8"
    )
    stopped = {
        "kind": "recorder_stopped",
        "telemetry_received": telemetry_count,
        "telemetry_enqueued": telemetry_count,
        "telemetry_written": telemetry_count,
        "telemetry_dropped": 0,
        "telemetry_write_errors": 0,
        "telemetry_pending": 0,
        "frame_encode_errors": 0,
        "finalize_errors": 0,
        "frames_skipped": 0,
        "frames_written": count,
        "events": 0,
    }
    (flight_dir / "events.jsonl").write_text(json.dumps(stopped) + "\n", encoding="utf-8")

    live_images: list[weakref.ReferenceType[np.ndarray]] = []
    peak_live = 0
    decode_calls = detect_calls = pose_calls = phase_calls = 0

    def sample() -> None:
        nonlocal peak_live
        peak_live = max(peak_live, sum(reference() is not None for reference in live_images))

    original_decode = cv2.imdecode
    original_detect = calibrate_module.detect_corners
    original_pose = calibrate_module.solve_board_pose
    from airdrop_calibration import algorithms as algorithms_module

    original_phase = algorithms_module._pattern_phase_quality

    def track_decode(encoded: np.ndarray, flags: int) -> np.ndarray | None:
        nonlocal decode_calls
        decode_calls += 1
        image = original_decode(encoded, flags)
        if image is not None:
            live_images.append(weakref.ref(image))
            sample()
        return image

    def track_detect(image: np.ndarray, pattern_size: tuple[int, int]) -> np.ndarray | None:
        nonlocal detect_calls
        detect_calls += 1
        sample()
        return original_detect(image, pattern_size)

    def track_pose(
        image: np.ndarray,
        corners: np.ndarray | None,
        object_points: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        *,
        pattern_size: tuple[int, int],
        square_size: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        nonlocal pose_calls
        pose_calls += 1
        sample()
        return original_pose(
            image,
            corners,
            object_points,
            camera_matrix,
            dist_coeffs,
            pattern_size=pattern_size,
            square_size=square_size,
        )

    def track_phase(
        image: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        *,
        pattern_size: tuple[int, int],
        square_size: float,
    ) -> tuple[float, float, int]:
        nonlocal phase_calls
        phase_calls += 1
        sample()
        return original_phase(
            image,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
            pattern_size=pattern_size,
            square_size=square_size,
        )

    monkeypatch.setattr(cv2, "imdecode", track_decode)
    monkeypatch.setattr(calibrate_module, "detect_corners", track_detect)
    monkeypatch.setattr(calibrate_module, "solve_board_pose", track_pose)
    monkeypatch.setattr(algorithms_module, "_pattern_phase_quality", track_phase)
    result = calibrate(flight_dir, pattern_size=ORIENTABLE_PATTERN)
    frame_meta = result["meta"]["frame_analysis"]
    assert frame_meta["analyzed_frames"] == count
    assert frame_meta["intrinsic_views"] == 24
    assert result["camera_matrix"] is not None
    assert decode_calls == count * 2 + 24  # 完整性、首遍检测、有效视图 PnP
    assert detect_calls == count
    assert pose_calls == 24
    assert phase_calls > 0
    assert peak_live <= 2


def test_calibration_rejects_same_size_image_changed_between_passes(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """第二遍不得把首遍角点与后来替换的同尺寸图像混合求解。"""
    flight_dir = workdir / "changed-between-passes"
    _write_calibration_flight(
        flight_dir, count=16, pattern_size=ORIENTABLE_PATTERN, position_available=False
    )
    original_solve = calibrate_module.run_intrinsics_from_corners

    def replace_after_intrinsics(observations, **kwargs):
        result = original_solve(observations, **kwargs)
        changed = np.full((TRUE_SIZE[1], TRUE_SIZE[0], 3), 240, dtype=np.uint8)
        assert cv2.imwrite(str(flight_dir / "frames" / "000000.jpg"), changed)
        return result

    monkeypatch.setattr(calibrate_module, "run_intrinsics_from_corners", replace_after_intrinsics)
    with pytest.raises(ValueError, match="两遍分析间图像像素内容改变"):
        calibrate(flight_dir, pattern_size=ORIENTABLE_PATTERN)


def test_calibrate_payload_records_lever_arm_against_measured_value(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """输出契约：``t_bc`` 恒为尺量值，平移估计进 ``meta`` 供对比。
    顺带钉住时间差的语义：素材里注入的是"视频延时 0.15 / 配置 lag 0.15 / 遥测延时 0"，
    所以标出来的必须是 **received 轴**的绝对量 ≈0.15（拿 capture 轴测会得到 ≈0.0）。
    """
    flight_dir = workdir / "flight"
    flight_dir.mkdir()
    measured = (0.05, -0.04, 0.08)
    monkeypatch.setattr(calibrate_module, "MEASURED_T_BC", measured)

    # 合成标定流程使用可由方格相位定向的棋盘尺寸。
    _write_calibration_flight(flight_dir, pattern_size=ORIENTABLE_PATTERN, position_available=False)
    payload = calibrate(flight_dir, pattern_size=ORIENTABLE_PATTERN, square_size=SQUARE)

    # 1) 标定文件里的 t_bc 恒为尺量值（georef 读的就是这个字段）
    assert payload["t_bc"] == list(measured)
    assert payload["meta"]["extrinsics"]["t_bc_source"] == "measured"
    arm = payload["meta"]["translation_diagnostic"]
    assert arm["measured"] == list(measured)
    assert payload["meta"]["extrinsics"]["mode"] == "rotation_only"
    assert arm["estimated"] is None
    # 2) 缺位置时只跳过独立对表，不改变旋转与延迟主结果。
    assert arm["status"] == "skipped"
    assert arm["estimated"] is None
    assert arm["reason"]
    # 3) 旋转仍以手眼标定为准
    assert rotation_angle_deg(np.asarray(payload["R_bc"]).T @ true_r_bc()) < 5.0
    # 4) 时间差：received 轴给出**绝对对齐量** = 视频延时 − 遥测延时 = 0.15 − 0 = 0.15
    assert payload["telemetry_lag"] == pytest.approx(0.15, abs=0.02)
    time_offset = payload["meta"]["time_offset"]
    assert time_offset["time_base"] == "received"
    assert time_offset["lag_baseline_s"] == pytest.approx(0.15)  # 素材里记录的录制基准
    # capture 轴残差 = 绝对量 − 基准 ≈ 0（两条轴互核通过）
    assert time_offset["lag_residual_on_capture_axis_s"] == pytest.approx(0.0, abs=0.02)
    assert time_offset["cross_check_ok"] is True


def test_translation_diagnostic_keeps_rotation_and_lag_independent(
    workdir: Path,
) -> None:
    flight_dir = workdir / "translation-flight"
    flight_dir.mkdir()
    _write_calibration_flight(flight_dir, pattern_size=ORIENTABLE_PATTERN, position_available=True)
    estimated = calibrate(flight_dir, pattern_size=ORIENTABLE_PATTERN, square_size=SQUARE)
    disabled = calibrate(
        flight_dir,
        pattern_size=ORIENTABLE_PATTERN,
        square_size=SQUARE,
        estimate_translation=False,
    )
    telemetry_path = flight_dir / "telemetry.jsonl"
    damaged = [json.loads(line) for line in telemetry_path.read_text(encoding="utf-8").splitlines()]
    for row in damaged:
        row["north_m"] = float("nan")
    telemetry_path.write_text(
        "\n".join(json.dumps(row) for row in damaged) + "\n", encoding="utf-8"
    )
    bad_position = calibrate(flight_dir, pattern_size=ORIENTABLE_PATTERN, square_size=SQUARE)
    arm = estimated["meta"]["translation_diagnostic"]
    assert arm["status"] == "estimated"
    assert arm["estimated"] is not None
    assert estimated["t_bc"] == disabled["t_bc"]
    assert np.allclose(estimated["R_bc"], disabled["R_bc"], atol=1e-11)
    assert estimated["telemetry_lag"] == pytest.approx(disabled["telemetry_lag"], abs=1e-11)
    assert disabled["meta"]["translation_diagnostic"]["status"] == "disabled"
    assert np.allclose(bad_position["R_bc"], estimated["R_bc"], atol=1e-11)
    assert bad_position["telemetry_lag"] == pytest.approx(estimated["telemetry_lag"], abs=1e-11)
    bad_arm = bad_position["meta"]["translation_diagnostic"]
    assert bad_arm["status"] == "skipped"
    assert bad_arm["estimated"] is None


@pytest.mark.parametrize("frame_encode_errors", [None, 1])
def test_full_capture_requires_zero_frame_encode_errors(
    workdir: Path, frame_encode_errors: int | None
) -> None:
    from airdrop_calibration.calibrate import _validate_full_update_capture

    flight_dir = workdir / "incomplete-capture"
    flight_dir.mkdir()
    (flight_dir / "config_snapshot.json").write_text(
        json.dumps({"record": {"telemetry_hz": 0}}), encoding="utf-8"
    )
    stopped = {
        "kind": "recorder_stopped",
        "telemetry_received": 0,
        "telemetry_enqueued": 0,
        "telemetry_written": 0,
        "telemetry_dropped": 0,
        "telemetry_write_errors": 0,
        "telemetry_pending": 0,
        "finalize_errors": 0,
        "frames_skipped": 0,
        "events": 0,
    }
    if frame_encode_errors is not None:
        stopped["frame_encode_errors"] = frame_encode_errors
    (flight_dir / "events.jsonl").write_text(json.dumps(stopped) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="遥测队列不完整"):
        _validate_full_update_capture(flight_dir, [])


def test_report_does_not_claim_extrinsics_when_hand_eye_failed() -> None:
    from airdrop_calibration.models import IntrinsicsResult, LagMeasurement, LagResult
    from airdrop_calibration.report import build_calibration_report

    payload = build_calibration_report(
        intrinsics=IntrinsicsResult(
            camera_matrix=np.eye(3),
            dist_coeffs=np.zeros(5),
            image_size=(640, 480),
            rms=0.2,
            views=10,
        ),
        lag=LagResult(lag_s=0.1, peak=0.9, samples=30),
        measurement=LagMeasurement(
            absolute=LagResult(lag_s=0.1, peak=0.9, samples=30),
            residual=None,
            baseline_s=0.0,
            cross_check_ok=True,
        ),
        extrinsics=None,
        flight_dir=Path("flight"),
        pattern_size=(8, 5),
        square_size=0.03,
        lag_search_s=0.5,
        lag_step_s=0.01,
        translation_diagnostic={
            "status": "skipped",
            "reason": "rotation_unavailable",
            "measured": list(MEASURED_T_BC),
            "estimated": None,
            "delta_xyz_m": None,
            "delta_m": None,
            "residual_m": None,
            "pairs": 0,
        },
    )
    assert payload["R_bc"] is None
    assert payload["meta"]["extrinsics"] is None
