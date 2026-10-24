"""相机标定的几何、内参、时间差和手眼计算。"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Callable, Sequence

import numpy as np

from .models import (
    ExtrinsicsResult,
    IntrinsicsResult,
    LagMeasurement,
    LagResult,
    TelemetryLike,
)
from .settings import (
    LAG_CROSS_CHECK_TOL_S,
    LAG_SEARCH_S,
    LAG_STEP_S,
    MAX_MOTION_PAIRS,
    MAX_VIEWS,
    MIN_AXIS_SPREAD_DEG,
    MIN_PAIR_ROTATION_DEG,
    MIN_VIEWS,
    PATTERN_SIZE,
    SQUARE_SIZE_M,
)


def rotation_x(angle_deg: float) -> np.ndarray:
    """返回绕 x 轴旋转的矩阵。"""
    angle = math.radians(angle_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=np.float64,
    )


def rotation_y(angle_deg: float) -> np.ndarray:
    """返回绕 y 轴旋转的矩阵。"""
    angle = math.radians(angle_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def rotation_z(angle_deg: float) -> np.ndarray:
    """返回绕 z 轴旋转的矩阵。"""
    angle = math.radians(angle_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def board_object_points(
    pattern_size: tuple[int, int] = PATTERN_SIZE,
    square_size: float = SQUARE_SIZE_M,
) -> np.ndarray:
    """棋盘格内角点的棋盘坐标系坐标（z=0 平面），形状 (N,1,3)。"""
    cols, rows = pattern_size
    points = np.zeros((cols * rows, 3), np.float64)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    points *= float(square_size)
    return points.reshape(-1, 1, 3)


def detect_corners(
    image: np.ndarray,
    pattern_size: tuple[int, int] = PATTERN_SIZE,
    *,
    refine: bool = True,
) -> np.ndarray | None:
    """找棋盘格内角点；找不到返回 None。

    先用 ``findChessboardCornersSB``（更稳、自带亚像素），失败再退回经典
    ``findChessboardCorners`` + ``cornerSubPix``。
    """
    import cv2  # 惰性导入：重依赖，只在真正做标定时加载（见模块 docstring）

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found = None
    if hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        ok, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=flags)
        if ok:
            found = corners.astype(np.float64).reshape(-1, 1, 2)
    if found is None:
        ok, corners = cv2.findChessboardCorners(
            gray,
            pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not ok:
            return None
        found = corners
        if refine:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
            found = cv2.cornerSubPix(gray, found, (11, 11), (-1, -1), criteria)
    return found.reshape(-1, 1, 2)


def pattern_phase_ok(
    image: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    *,
    pattern_size: tuple[int, int] = PATTERN_SIZE,
    square_size: float = SQUARE_SIZE_M,
) -> float:
    """用内侧方格的黑白相位校验位姿，返回相符比例（0~1）。

    用途只有一个：**发现"棋盘格整体错格"这类几何错误**。例如 :func:`render_board`
    的 ``to_board`` 少减一个 ``tile`` 时，棋盘格相对对象点整体错开一整格，
    此时按给定位姿把方格中心投影回图里，落到的是**反相**的格子，相符率会掉到 0.5 以下
    （实测正确渲染是 1.000）。重投影误差对这类错误**完全不敏感**（仍有 0.1 px），
    只有相位能发现。

    该分数只反映采样相符比例。候选位姿筛选另要求黑白中位数对比度至少 25 灰阶，
    并要求足够数量的有效内侧方格样本，避免低对比图像被噪声误判为高分。
    """
    return _pattern_phase_quality(
        image,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
        pattern_size=pattern_size,
        square_size=square_size,
    )[0]


def _pattern_phase_quality(
    image: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    *,
    pattern_size: tuple[int, int],
    square_size: float,
) -> tuple[float, float, int]:
    """返回相位相符比例、黑白对比度和有效内侧方格数。"""
    import cv2

    cols, rows = pattern_size
    centres = [
        ((col + 0.5) * square_size, (row + 0.5) * square_size, 0.0)
        for row in range(rows - 1)
        for col in range(cols - 1)
    ]
    want_light = [(col + row) % 2 == 0 for row in range(rows - 1) for col in range(cols - 1)]
    points = np.asarray(centres, dtype=np.float64).reshape(-1, 1, 3)
    projected, _ = cv2.projectPoints(points, rvec, tvec, camera_matrix, dist_coeffs)
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    samples: list[tuple[float, bool]] = []
    for (u, v), expect_light in zip(projected.reshape(-1, 2), want_light, strict=True):
        x, y = int(round(float(u))), int(round(float(v)))
        if x < 3 or y < 3 or x >= gray.shape[1] - 4 or y >= gray.shape[0] - 4:
            continue
        patch = gray[y - 3 : y + 4, x - 3 : x + 4]
        samples.append((float(np.median(patch)), expect_light))
    bright = [value for value, light in samples if light]
    dark = [value for value, light in samples if not light]
    if len(samples) < 8 or min(len(bright), len(dark)) < 4:
        return 0.0, 0.0, len(samples)
    contrast = float(np.median(bright) - np.median(dark))
    threshold = (float(np.median(bright)) + float(np.median(dark))) / 2
    score = sum(value > threshold if light else value < threshold for value, light in samples)
    return score / len(samples), contrast, len(samples)


def solve_board_poses(
    images: Sequence[np.ndarray],
    object_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    *,
    pattern_size: tuple[int, int] = PATTERN_SIZE,
    square_size: float = SQUARE_SIZE_M,
) -> list[tuple[np.ndarray, np.ndarray] | None]:
    """逐帧检测 + 解位姿；用黑白格心相位选择唯一角点方向。

    内角点列/行同奇偶时，反序映射不改变内部方格相位，单帧图像无法选择方向，故拒绝。
    奇偶相反时，反序会翻转相位；只接受样本足够、对比度足够且恰有一个候选匹配的帧。

    返回与 ``images`` 等长的列表，检不出角点的位置为 ``None``。
    """
    import cv2  # 惰性导入：重依赖，只在真正做标定时加载（见模块 docstring）

    cols, rows = pattern_size
    if (cols - rows) % 2 == 0:
        raise ValueError("棋盘格内角点列数与行数必须一奇一偶，才能用单帧黑白相位判定方向")
    poses: list[tuple[np.ndarray, np.ndarray] | None] = []
    for image in images:
        corners = detect_corners(image, pattern_size)
        if corners is None:
            poses.append(None)
            continue
        candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
        for candidate_corners in (corners, corners[::-1]):
            ok, rvec, tvec = cv2.solvePnP(
                object_points,
                candidate_corners,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE,
            )
            if ok:
                score, contrast, samples = _pattern_phase_quality(
                    image,
                    rvec,
                    tvec,
                    camera_matrix,
                    dist_coeffs,
                    pattern_size=pattern_size,
                    square_size=square_size,
                )
                if samples >= 8 and contrast >= 25.0:
                    candidates.append((score, rvec, tvec))
        if not candidates:
            poses.append(None)
            continue
        candidates.sort(key=lambda item: item[0], reverse=True)
        if candidates[0][0] < 0.8 or (
            len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.5
        ):
            poses.append(None)
            continue
        poses.append((candidates[0][1], candidates[0][2]))
    return poses


def render_board(
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_size: tuple[int, int],
    *,
    pattern_size: tuple[int, int] = PATTERN_SIZE,
    square_size: float = SQUARE_SIZE_M,
    pixels_per_square: int = 70,
    margin: int = 40,
    background: int = 20,
    tile_value: int = 235,
) -> np.ndarray:
    """把棋盘格按给定位姿渲染成一张图（用于标定流水线的合成自洽验证）。

    实现方式：在棋盘坐标系里先做一次正投影得到"正视模板"，再用
    ``cv2.getPerspectiveTransform`` 把模板的四角映射到真实位姿下的四角——
    对 z=0 平面来说这就是精确的透视变换（不是近似）。

    **配色必须是"深底 + 亮格"（左上角外格为亮格）**：``findChessboardCorners`` 依赖
    标准棋盘格的明暗极性，反了会直接检不出（实测：极性反了 SB 与经典版
    双双返回 False，而 ``checkChessboard`` 仍为 True——说明几何没问题、只是极性）。
    """
    import cv2  # 惰性导入：重依赖，只在真正做标定时加载（见模块 docstring）

    cols, rows = pattern_size
    tile = int(pixels_per_square)
    # 模板**不带外框边距**：模板边界就是棋盘的物理边界（(cols+1)×(rows+1) 个格子）。
    # 早期版本在四周留了 margin，结果背景与边格同色，"棋盘边界"在视觉上不可辨
    # （图上边一排亮格像是凸出来的标签），findChessboardCorners 直接检不出——
    # 标准棋盘必须是干净的矩形边界。
    template_w = (cols + 1) * tile
    template_h = (rows + 1) * tile
    template = np.full((template_h, template_w), background, np.uint8)
    for row in range(rows + 1):
        for col in range(cols + 1):
            if (row + col) % 2:
                continue
            cv2.rectangle(
                template,
                (col * tile, row * tile),
                (
                    min((col + 1) * tile - 1, template_w - 1),
                    min((row + 1) * tile - 1, template_h - 1),
                ),
                tile_value,
                -1,
            )

    # 模板四角在棋盘坐标系里的位置。
    # ⚠ **(0,0) 对应的是棋盘的第一个"内角点"，不是模板左上角像素**：一格 = 一个方格，
    # 而内角点在格子交点上，所以模板像素 (tile, tile) 才是棋盘坐标 (0,0)。
    # 少了这个 −tile 偏移，渲染出来的棋盘格会整体错开**整整一格**（方格黑白还整体反相），
    # 检测出的角点落在"板子真实点 +1 格"处，PnP 只能靠平面二义性去凑一个 180° 的解——
    # 实测正面视图误差 248 px、姿态差精确 180°，而重投影误差仍只有 0.19 px（极难察觉）。
    def to_board(px: float, py: float) -> tuple[float, float, float]:
        return (
            (px - tile) / tile * square_size,
            (py - tile) / tile * square_size,
            0.0,
        )

    template_corners = np.array(
        [
            to_board(0, 0),
            to_board(template.shape[1] - 1, 0),
            to_board(template.shape[1] - 1, template.shape[0] - 1),
            to_board(0, template.shape[0] - 1),
        ],
        dtype=np.float64,
    )
    projected, _ = cv2.projectPoints(template_corners, rvec, tvec, camera_matrix, dist_coeffs)
    src = np.array(
        [
            [0.0, 0.0],
            [template.shape[1] - 1.0, 0.0],
            [template.shape[1] - 1.0, template.shape[0] - 1.0],
            [0.0, template.shape[0] - 1.0],
        ],
        dtype=np.float32,
    )
    dst = projected.reshape(-1, 2).astype(np.float32)
    # 目标图要足够大以容纳投影后的板子
    canvas_w = max(int(image_size[0]), int(np.abs(dst[:, 0]).max()) + 10)
    canvas_h = max(int(image_size[1]), int(np.abs(dst[:, 1]).max()) + 10)
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        template,
        matrix,
        (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=background,
    )
    # 裁到目标画幅
    canvas = np.full((image_size[1], image_size[0]), background, np.uint8)
    h = min(image_size[1], warped.shape[0])
    w = min(image_size[0], warped.shape[1])
    canvas[:h, :w] = warped[:h, :w]
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def look_at_board(
    distance: float,
    camera_matrix: np.ndarray,
    *,
    tilt_deg: float = 0.0,
    yaw_deg: float = 0.0,
    rim_roll_deg: float = 0.0,
    target_px: tuple[float, float] | None = None,
    pattern_size: tuple[int, int] = PATTERN_SIZE,
    square_size: float = SQUARE_SIZE_M,
) -> tuple[np.ndarray, np.ndarray]:
    """让相机正对棋盘格中心（返回相机在棋盘系下的 rvec/tvec）。

    棋盘在 z=0 平面上、**内角点**范围是 ``[0,(cols-1)·s] × [0,(rows-1)·s]``，
    板子几何中心是 ``((cols-1)·s/2, (rows-1)·s/2, 0)``——不是原点。

    位姿构造方式：取"相机在板子正上方、相机 z 轴指向板子"这个基准位姿，
    再用绕**相机自身轴**的旋转叠加 tilt/yaw/rim_roll 扰动（标定需要的姿态
    多样性）。**基准位姿矩阵不手推**，而是构造一个已知能投影到目标点的相机模型
    来取得——手推旋转矩阵经历过两次约定错误（一次把相机轴当行/列搞反，
    一次 180° 绕错轴），不值得再来第三次。

    ⚠ 扰动是**旋转矩阵右乘**，不是把角度加在旋转向量上：旋转向量只在零附近才近似
    线性，基准位姿的 ``|rvec|`` 一旦接近 π（例如``v`` 轴反向的写法就会得到 π），
    "向量相加"会给出与预期完全不同的姿态，而且症状是"时好时坏"，极难定位。

    返回的 ``(rvec, tvec)`` 由 ``cv2.solvePnP`` 在"棋盘格角点 + 这些相机参数"
    上解出，因此**保证与 OpenCV 的投影约定一致**。
    """
    import cv2  # 惰性导入：重依赖，只在真正做标定时加载（见模块 docstring）

    cols, rows = pattern_size
    center = np.array([(cols - 1) * square_size / 2.0, (rows - 1) * square_size / 2.0, 0.0])
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    if target_px is None:
        target_px = (float(camera_matrix[0, 2]), float(camera_matrix[1, 2]))
    base = _base_look_down_pose(
        distance,
        camera_matrix,
        target_px,
        center,
        pattern_size=pattern_size,
        square_size=square_size,
    )
    disturbance = np.radians(np.array([rim_roll_deg, tilt_deg, yaw_deg], dtype=np.float64))
    rotation = cv2.Rodrigues(base[0].reshape(3, 1))[0] @ cv2.Rodrigues(disturbance.reshape(3, 1))[0]
    return cv2.Rodrigues(rotation)[0], base[1]


def _base_look_down_pose(
    distance: float,
    camera_matrix: np.ndarray,
    target_px: tuple[float, float],
    board_center: np.ndarray,
    *,
    pattern_size: tuple[int, int] = PATTERN_SIZE,
    square_size: float = SQUARE_SIZE_M,
) -> tuple[np.ndarray, np.ndarray]:
    """构造"相机在板子正上方、光轴竖直向下、**看板子正面**"的位姿，用 solvePnP 反解 rvec/tvec。

    直接手推旋转矩阵容易在约定上出错；这里改用"造一个临时相机模型 → solvePnP"
    的办法，让 OpenCV 自己给出与它投影约定一致的位姿。

    ⚠ 图像 v 轴与板子 +y **同向**（``v = cy + dy·scale``）是经过实测的：反过来会让
    合成图上的棋盘格成为**镜像**，检测出的角点标签整体反序，PnP 于是稳定地落到
    另一个分支上（实测 4 个视图全部差 180°）。别"顺手改成更直观的"写法。
    """
    import cv2  # 惰性导入：重依赖，只在真正做标定时加载（见模块 docstring）

    cols, rows = pattern_size
    # 临时相机：主点设在希望板子中心落在的位置，光轴竖直向下（俯视）
    temp = camera_matrix.copy()
    temp[0, 2] = float(target_px[0])
    temp[1, 2] = float(target_px[1])
    board = board_object_points(pattern_size, square_size)
    # 临时视图：板子四角在临时相机下的投影（俯视 → 正视缩放）
    focal = (float(temp[0, 0]) + float(temp[1, 1])) / 2.0
    scale = focal / float(distance)  # 米 → 像素
    image_points = np.zeros((cols * rows, 1, 2), np.float64)
    for index in range(cols * rows):
        bx, by, _ = board[index, 0]
        dx = bx - float(board_center[0])
        dy = by - float(board_center[1])
        # 图像 u 随板子 +x 增大、v 随板子 +y 增大（与模板像素轴一致，保证不镜像）
        image_points[index, 0, 0] = temp[0, 2] + dx * scale
        image_points[index, 0, 1] = temp[1, 2] + dy * scale
    ok, rvec, tvec = cv2.solvePnP(
        board, image_points, temp, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        raise RuntimeError("无法构造棋盘格基准位姿（solvePnP 失败）")
    return rvec.reshape(3), tvec.reshape(3, 1)


def run_intrinsics(
    frames: Sequence[np.ndarray],
    *,
    pattern_size: tuple[int, int] = PATTERN_SIZE,
    square_size: float = SQUARE_SIZE_M,
    min_views: int = MIN_VIEWS,
    max_views: int = MAX_VIEWS,
) -> IntrinsicsResult:
    """棋盘格多视图内参标定。

    ⚠ **OpenCV 5.0 的 ``calibrateCamera`` 只收 float32 的 objectPoints/imagePoints**
    （实测：任一为 float64 都抛 ``objectPoints should contain vector of vectors of
    points of type Point3f``）。而 :func:`detect_corners` 的 SB 路径给的是 float64，
    所以这里必须显式转 float32——不转的话内参这步会直接失败。
    """
    import cv2  # 惰性导入：重依赖，只在真正做标定时加载（见模块 docstring）

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    size: tuple[int, int] | None = None
    board = board_object_points(pattern_size, square_size).astype(np.float32)

    for frame in frames:
        if size is None:
            size = (int(frame.shape[1]), int(frame.shape[0]))
        corners = detect_corners(frame, pattern_size)
        if corners is None:
            continue
        object_points.append(board.copy())
        image_points.append(corners.astype(np.float32))
        if len(object_points) >= max_views:
            break

    if size is None:
        raise ValueError("没有任何图像帧")
    if len(object_points) < max(3, min_views):
        raise ValueError(
            f"有效视图不足：找到 {len(object_points)} 个（至少需要 {max(3, min_views)} 个）。"
            "检查棋盘格内角点数 PATTERN_SIZE 是否与实际板子一致，以及棋盘格是否全程可见"
        )

    rms, matrix, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, size, None, None
    )
    per_view: list[float] = []
    for index, (obj, img) in enumerate(zip(object_points, image_points, strict=True)):
        projected, _ = cv2.projectPoints(obj, rvecs[index], tvecs[index], matrix, dist)
        error = float(np.sqrt(np.mean(np.sum((projected - img) ** 2, axis=2))))
        per_view.append(error)
    return IntrinsicsResult(
        camera_matrix=np.asarray(matrix, dtype=np.float64),
        dist_coeffs=np.asarray(dist, dtype=np.float64).reshape(-1),
        image_size=size,
        rms=float(rms),
        views=len(object_points),
        per_view_errors=per_view,
    )


def rotation_angle_deg(rotation: np.ndarray) -> float:
    """旋转矩阵对应的旋转角（度）。"""
    trace = float(np.trace(rotation))
    cos_theta = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return math.degrees(math.acos(cos_theta))


def quaternion_to_matrix_local(
    quaternion: tuple[float, float, float, float],
) -> np.ndarray:
    """四元数 (w,x,y,z) → 旋转矩阵（与 georef.quaternion_to_matrix 同一约定）。"""
    values = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if norm < 1e-12:
        raise ValueError("零四元数不能表示旋转")
    w, x, y, z = values / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def angular_speed_from_poses(
    timestamps: Sequence[float], rotations: Sequence[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """由逐帧旋转序列算角速度模长（度/秒）。

    返回 ``(中间时刻, 角速度)``。刚体角速度的**模长与安装角无关**，
    所以这一步可以完全在外参未知的情况下做——这正是它能先于外参的原因。
    """
    times: list[float] = []
    speeds: list[float] = []
    for index in range(1, len(rotations)):
        dt = float(timestamps[index] - timestamps[index - 1])
        if dt <= 0:
            continue
        delta = rotations[index - 1].T @ rotations[index]
        angle = rotation_angle_deg(delta)
        times.append(0.5 * (timestamps[index] + timestamps[index - 1]))
        speeds.append(angle / dt)
    return np.asarray(times), np.asarray(speeds, dtype=np.float64)


def estimate_lag_by_correlation(
    camera_times: np.ndarray,
    camera_speeds: np.ndarray,
    body_times: np.ndarray,
    body_speeds: np.ndarray,
    *,
    search_s: float = LAG_SEARCH_S,
    step_s: float = LAG_STEP_S,
) -> LagResult:
    """互相关求"画面比遥测晚多少秒"。

    把画面角速度序列按 ``+lag`` 平移后再与机体系列比对，取相关系数峰值；
    峰值附近做抛物线插值取亚样本精度。``lag > 0`` 表示画面比姿态**滞后**。
    """
    if camera_speeds.size < 3 or body_speeds.size < 3:
        return LagResult(lag_s=float("nan"), peak=float("nan"), samples=0)

    base = min(float(camera_times[0]), float(body_times[0]))
    cam_t = camera_times - base
    body_t = body_times - base

    # 重采样到统一时间基（线性插值，超出范围用端点值而不是外推）
    def resample(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
        return np.interp(grid, times, values)

    n = max(16, min(2000, int(max(cam_t[-1], body_t[-1]) / max(step_s, 1e-3)) + 1))
    grid = np.linspace(0.0, max(cam_t[-1], body_t[-1]), n)
    cam = resample(cam_t, camera_speeds, grid)
    body = resample(body_t, body_speeds, grid)

    cam_centered = cam - cam.mean()
    body_centered = body - body.mean()
    cam_norm = float(np.linalg.norm(cam_centered))
    body_norm = float(np.linalg.norm(body_centered))
    if cam_norm < 1e-9 or body_norm < 1e-9:
        return LagResult(lag_s=float("nan"), peak=float("nan"), samples=0)

    shifts = np.arange(-search_s, search_s + 1e-12, step_s)
    # 网格点之间的时间间隔
    dt_grid = float(grid[1] - grid[0]) if n > 1 else step_s
    curve: list[tuple[float, float]] = []
    correlations: list[float] = []
    for lag in shifts:
        # 画面滞后 lag → 把画面序列前移 lag/grid 个样本与机体对齐
        offset = int(round(lag / dt_grid))
        if offset == 0:
            shifted = cam_centered
        elif offset > 0:
            shifted = np.concatenate([cam_centered[offset:], np.zeros(offset)])
        else:
            shifted = np.concatenate([np.zeros(-offset), cam_centered[:offset]])
        denominator = cam_norm * body_norm
        value = float(np.dot(shifted, body_centered) / denominator) if denominator else 0.0
        correlations.append(value)
        curve.append((float(lag), value))

    values = np.asarray(correlations)
    best = int(np.argmax(values))
    lag = float(shifts[best])
    # 抛物线亚样本插值
    if 0 < best < len(values) - 1:
        left, middle, right = values[best - 1], values[best], values[best + 1]
        denom = left - 2.0 * middle + right
        if abs(denom) > 1e-12:
            lag += 0.5 * (left - right) / denom * step_s
    # ⚠ 收窄成 Python float：numpy 标量既过不了 JSON 序列化，也让 `is True` 类断言失效
    lag = float(lag)
    return LagResult(
        lag_s=lag,
        peak=float(values[best]),
        samples=int(camera_speeds.size),
        curve=curve,
    )


def measure_frame_telemetry_lag(
    received_times: Sequence[float],
    capture_times: Sequence[float],
    board_rotations: Sequence[np.ndarray],
    snapshots: Sequence[TelemetryLike],
    *,
    baseline_s: float,
    search_s: float = LAG_SEARCH_S,
    step_s: float = LAG_STEP_S,
) -> LagMeasurement:
    """第 2 步：同一批棋盘格旋转，在**两条时间轴**上各测一次，并互核。

    为什么必须用 received 轴（这是本项目踩过的坑）
    ---------------------------------------------
    ``capture_times`` 是录制时**按配置的 ``telemetry_lag`` 假设出来的**拍摄时刻
    （``= received − lag``，见 :class:`FrameStamp`）。拿它去测时间差，测出来的是
    "该假设的残差"：**换一次配置，结果就跟着平移一个 lag**——真机上表现为"回填上一次
    标定值再标一次，Δt 自己收敛到 0"，但那不代表标定对了，只代表配置与假设一致了。
    用 ``received_times``（原始量）测出来的才是**绝对对齐量**：

        ``绝对对齐量 = 视频链路延时 d_video − 四元数接收参照延时 d_quaternion``

    ``snapshots`` 必须是已按四元数原始接收时刻去重的姿态序列，生产入口由
    ``SourceTimeline.attitudes`` 提供。合并快照组装时刻不是该参照，直接传入会测错。
    第 3 步和运行时都在 ``received_timestamp − telemetry_lag`` 查询源时标对齐位姿。
    这个查询时刻是四元数接收参照上的时刻，不是精确曝光墙钟。
    受控实验（固定 d_video/d_tel、只改录制时配置的 lag）：received 轴的结果不变、
    capture 轴的结果恰好平移一个 lag；详见 ``docs/calibration_opencv.md``。

    互核
    ----
    ``baseline_s + residual ≈ absolute``（两条轴只差一个基准），
    对不上说明素材的记账方式与预期不符（例如索引里的 ``lag`` 与录制配置不一致），
    只告警、不改结果——绝对量仍以 received 轴为准。
    """
    # ⚠ 时刻与姿态必须**同一个筛选条件**取出来：两者分开筛会静默错配（四元数缺几条时）
    paired_body = [
        (item.timestamp, quaternion_to_matrix_local(quaternion))
        for item in snapshots
        if (quaternion := _quaternion_of(item)) is not None
    ]
    body_times, body_speeds = angular_speed_from_poses(
        [stamp for stamp, _ in paired_body],
        [rotation for _, rotation in paired_body],
    )
    received_t, received_w = angular_speed_from_poses(received_times, board_rotations)
    absolute = estimate_lag_by_correlation(
        received_t, received_w, body_times, body_speeds, search_s=search_s, step_s=step_s
    )
    residual: LagResult | None = None
    cross_check_ok = False
    if capture_times:
        capture_t, capture_w = angular_speed_from_poses(capture_times, board_rotations)
        residual = estimate_lag_by_correlation(
            capture_t, capture_w, body_times, body_speeds, search_s=search_s, step_s=step_s
        )
        if absolute.ok and residual.ok:
            cross_check_ok = bool(
                abs(baseline_s + residual.lag_s - absolute.lag_s) <= LAG_CROSS_CHECK_TOL_S
            )
    return LagMeasurement(
        absolute=absolute,
        residual=residual,
        baseline_s=float(baseline_s),
        cross_check_ok=cross_check_ok,
    )


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = float(vector[0]), float(vector[1]), float(vector[2])
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rotation_log(matrix: np.ndarray) -> np.ndarray:
    """旋转矩阵 → 轴角向量。"""
    trace = float(np.trace(matrix))
    cos_theta = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    theta = math.acos(cos_theta)
    if theta < 1e-12:
        return np.zeros(3)
    if abs(math.pi - theta) < 1e-6:
        diag = np.clip((np.diag(matrix) + 1.0) / 2.0, 0.0, None)
        axis = np.sqrt(diag)
        index = int(np.argmax(axis))
        if axis[index] > 1e-12:
            others = [(index + 1) % 3, (index + 2) % 3]
            for other in others:
                axis[other] = matrix[index, other] / (2.0 * axis[index])
        norm = float(np.linalg.norm(axis))
        return axis / norm * theta if norm > 1e-12 else np.zeros(3)
    axis = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ]
    ) / (2.0 * math.sin(theta))
    return axis * theta


def solve_hand_eye(
    a_list: Sequence[np.ndarray],
    b_list: Sequence[np.ndarray],
    *,
    use_translation: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, float, float]:
    """求解 ``A X = X B``（与 ``cv::calibrateHandEye`` 同一约定）。

    A、B 的来源是唯一正确的配对，别换
    ----------------------------------
    * ``A``：机体在**世界系**里的相对运动，来自**遥测位姿** ``G = T_world_body``::

          A_ij = G_j⁻¹ · G_i

    * ``B``：棋盘格在**相机系**里的相对运动，来自 **PnP** ``P = T_cam_board``::

          B_ij = P_j · P_i⁻¹

    待求 ``X = T_body_cam = [R_bc | t_bc]``，与 :mod:`airdrop.georef` 的
    ``R_bc``（相机系→机体系）、``t_bc``（光心在机体系）**同一语义**。

    推导：棋盘格固定在世界里（``C = T_world_board`` 为常量），于是
    ``P_i = X⁻¹ G_i⁻¹ C``；代入两帧得 ``B_ij = X⁻¹ A_ij X``，即 ``A X = X B``。

    ⚠ 退化陷阱（症状是"稳定解出单位矩阵"）
    ------------------------------------------------------
    若把"机体位姿"写成**由棋盘格位姿左乘一个常量**导出的量（``T_body = X · P``，
    哪怕左乘的就是真值），相对运动里那个常量会被整体消掉::

        A = (X P_i)⁻¹ (X P_j) = P_i⁻¹ P_j ≡ B

    于是 ``R_a ≡ R_b``，方程退化成"求与所有 ``R_a`` 可交换的矩阵"，唯一解是单位矩阵，
    **真值根本不在解空间里**。这跟"板子有没有转"无关——板子固定、飞机在动时棋盘格
    相对机体的姿态当然逐帧在变；根因是**拿未知量当已知量用**。
    正确做法只有一条：``A`` 必须来自**独立的遥测位姿**。
    回归用例 ``test_hand_eye_degenerate_when_body_derived_from_camera`` 把它钉死。

    旋转（Horaud 向量化）
    --------------------
    ``R_a R_x = R_x R_b``。按**列优先** vec 的两个恒等式
    ``vec(R_a X) = (I ⊗ R_a) vec(X)``、``vec(X R_b) = (R_bᵀ ⊗ I) vec(X)`` 得::

        [ (I ⊗ R_a) − (R_bᵀ ⊗ I) ] · vec(R_x) = 0

    堆叠全部运动对后取最小奇异值对应的**右奇异向量**（``vt[-1]``：``np.linalg.svd``
    返回的 ``vt`` 每一**行**才是右奇异向量），再正交化。

    ⚠ **必须按 F 序拆回 3×3**：``reshape(3, 3)`` 默认是 C 序，会给出另一个"看起来也像
    合法旋转矩阵"的错解，肉眼分辨不出来。

    平移
    ----
    ``R_a t_x + t_a = R_x t_b + t_x`` ⇒ ``(R_a − I) t_x = R_x t_b − t_a``。
    位置平移由独立的 :func:`estimate_translation_fixed_rotation` 在固定已求旋转后估计，
    其结果与尺量值一并输出供人工判断。

    平移方程按所有运动对堆叠为 ``(3n×3)`` 最小二乘。

    返回 ``(R_x, t_x | None, 相对奇异值, 旋转轴张角(度))``。
    相对奇异值 ``s[-1]/s[0]`` 越接近 0 说明运动激励越充分（绝对值随量纲变化，只有相对量可比）。
    """
    if len(a_list) != len(b_list) or len(a_list) < 2:
        raise ValueError(f"运动对数量不足: {len(a_list)}")
    rows = []
    axes: list[np.ndarray] = []
    for a_matrix, b_matrix in zip(a_list, b_list, strict=True):
        rotation_a = a_matrix[:3, :3]
        rotation_b = b_matrix[:3, :3]
        rows.append(np.kron(np.eye(3), rotation_a) - np.kron(rotation_b.T, np.eye(3)))
        axes.append(_rotation_axis(rotation_a))
    stacked = np.vstack(rows)
    _, singular, vt = np.linalg.svd(stacked)
    relative = float(singular[-1] / singular[0]) if singular[0] > 0 else float("inf")
    candidate = vt[-1].reshape(3, 3, order="F")
    # ⚠ 零空间向量只定到"差一个符号"，而 ``vec(R)`` 与 ``vec(−R)`` 表示**同一个旋转**。
    # 但 ``−R`` 的行列式是 −1、不是旋转矩阵：若直接把它喂给"翻转最小奇异值"式的正交化，
    # 会得到 ``R·diag(−1,−1,1)``——**恰好是绕 z 的 180° 旋转**。症状极其阴险：
    # 零空间确实是一维、真值确实在里面，解出来却精确偏 180°，而且**时对时错**
    # （取决于 SVD 给出的符号）。正确做法是先按行列式**整体取反**，再做正交化。
    if float(np.linalg.det(candidate)) < 0.0:
        candidate = -candidate
    u, _, vt2 = np.linalg.svd(candidate)
    rotation = u @ np.diag([1.0, 1.0, float(np.linalg.det(u @ vt2))]) @ vt2

    translation: np.ndarray | None = None
    if use_translation:
        # 堆叠成 (3n×3) 最小二乘（= 法方程 AᵀA t = Aᵀb）。
        # ⚠ 别改回"逐行相加再解 3×3"：那样等于给每对**等权**，旋转角小的对没有信息却带进噪声；
        # 实测杆臂误差中位大 1.4–3.5 倍，且与 cv2 的 PARK 方法（就是本法）不再逐位一致。
        d_rows = np.vstack([a_matrix[:3, :3] - np.eye(3) for a_matrix in a_list])
        d_rhs = np.concatenate(
            [
                rotation @ b_matrix[:3, 3] - a_matrix[:3, 3]
                for a_matrix, b_matrix in zip(a_list, b_list, strict=True)
            ]
        )
        translation, *_ = np.linalg.lstsq(d_rows, d_rhs, rcond=None)
    return rotation, translation, relative, _axis_spread_deg(axes)


def _rotation_axis(rotation: np.ndarray) -> np.ndarray:
    """旋转矩阵的转轴（单位向量）；无旋转返回零向量。"""
    vector = rotation_log(rotation)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else np.zeros(3)


def _motion_pairs(count: int, max_pairs: int) -> list[tuple[int, int]]:
    if max_pairs < 1:
        raise ValueError("max_pairs 必须大于零")
    pairs = [(index, other) for index in range(count) for other in range(index + 1, count)]
    return pairs[:: len(pairs) // max_pairs + 1] if len(pairs) > max_pairs else pairs


def _axis_spread_deg(axes: Sequence[np.ndarray]) -> float:
    """一组转轴之间的最大夹角（度）。转轴是双向的，所以正负号不影响结果。

    ⚠ 原来写成纯 Python 双重循环。``axes`` 的长度是通过 ``min_pair_rotation_deg``
    过滤后剩下的运动对数，上限 ``MAX_MOTION_PAIRS = 2000``：200 个位姿 → 抽样后
    约 1990 对 → 近 200 万次内层迭代，每次含 2 次 ``np.linalg.norm`` + 1 次
    ``np.dot`` + ``acos``，按 ~6µs 算就是十几秒，而它在每次 ``solve_hand_eye``
    里都无条件跑。这条判据本身是廉价的安全检查，不该是热点。

    改成一次性向量化：归一化 → Gram 矩阵 → 取上三角 → arccos 取最大值。
    数学上与原来逐对算的完全一致（同样的 ``|dot| / (|a||b|)``）。
    """
    if len(axes) < 2:
        return 0.0
    vectors = np.asarray(axes, dtype=np.float64).reshape(-1, 3)
    norms = np.linalg.norm(vectors, axis=1)
    keep = norms > 1e-12
    vectors, norms = vectors[keep], norms[keep]
    if len(vectors) < 2:
        return 0.0
    unit = vectors / norms[:, None]
    gram = np.abs(unit @ unit.T)
    upper = np.triu_indices(len(unit), k=1)
    return float(np.degrees(np.arccos(np.clip(gram[upper], -1.0, 1.0))).max())


def relative_transform(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """相对位姿 ``T_a^{-1} T_b``（输入为 4×4）。"""
    return np.linalg.inv(a) @ b


def _quaternion_of(snapshot: TelemetryLike) -> tuple[float, float, float, float] | None:
    """快照里的姿态四元数 ``(w, x, y, z)``；任一分量缺失就返回 None（**不拿 0 顶上**）。"""
    w, x, y, z = (
        snapshot.quaternion_w,
        snapshot.quaternion_x,
        snapshot.quaternion_y,
        snapshot.quaternion_z,
    )
    if w is None or x is None or y is None or z is None:
        return None
    return (float(w), float(x), float(y), float(z))


def body_pose_matrix(snapshot: TelemetryLike) -> np.ndarray | None:
    """遥测快照 → ``T_world_body``（4×4）。位置或姿态缺一不可，否则返回 None。"""
    if (
        snapshot.north_m is None
        or snapshot.east_m is None
        or snapshot.down_m is None
        or snapshot.quaternion_w is None
        or snapshot.quaternion_x is None
        or snapshot.quaternion_y is None
        or snapshot.quaternion_z is None
    ):
        return None
    matrix = np.eye(4)
    matrix[:3, :3] = quaternion_to_matrix_local(
        (
            float(snapshot.quaternion_w),
            float(snapshot.quaternion_x),
            float(snapshot.quaternion_y),
            float(snapshot.quaternion_z),
        )
    )
    matrix[:3, 3] = (snapshot.north_m, snapshot.east_m, snapshot.down_m)
    return matrix


def _slerp(first: Sequence[float], second: Sequence[float], weight: float) -> np.ndarray:
    """四元数球面插值（w 在前，含 q/−q 翻转）。"""
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    if dot > 0.9995:  # 夹角太小：线性插值再归一化更稳
        return (1.0 - weight) * a + weight * b
    theta = math.acos(min(1.0, max(-1.0, dot)))
    return (math.sin((1.0 - weight) * theta) * a + math.sin(weight * theta) * b) / math.sin(theta)


def telemetry_pose_at(snapshots: Sequence[TelemetryLike], timestamp: float) -> np.ndarray | None:
    """取 ``timestamp`` 时刻的 ``T_world_body``：位置线性内插、姿态 slerp。

    只做**范围内**内插；时刻落在遥测覆盖之外返回 None（不外推——标定不能被外推出来的
    姿态污染）。``snapshots`` 需按 ``timestamp`` 升序（``FlightLog.iter_telemetry`` 即升序）。
    """
    return _telemetry_pose_from_index(_index_telemetry(snapshots), timestamp)


def _index_telemetry(
    snapshots: Sequence[TelemetryLike],
) -> tuple[list[float], list[TelemetryLike], list[np.ndarray]]:
    """把遥测序列整成 ``(times, usable, poses)``，**只算一次**。

    ⚠ 原来每次调用都重建这三样：``[item for item in snapshots if
    body_pose_matrix(item) is not None]`` 会把**全部**快照过一遍，而
    ``body_pose_matrix`` 每次都 ``np.eye(4)`` + 一次 ``quaternion_to_matrix``。
    ``align_body_poses`` 对每个棋盘格位姿调一次本函数——采集默认 180s、遥测按
    ``record.telemetry_hz=10`` 落盘约 1800 条，200 个位姿就是 **36 万次**
    ``quaternion_to_matrix`` + 36 万次 ``np.eye(4)``，几秒纯浪费。

    现在由 :func:`align_body_poses` 建一次并传下来。
    """
    usable: list[TelemetryLike] = []
    poses: list[np.ndarray] = []
    for item in snapshots:
        pose = body_pose_matrix(item)
        if pose is None:
            continue
        usable.append(item)
        poses.append(pose)
    return [item.timestamp for item in usable], usable, poses


def _telemetry_pose_from_index(
    index: tuple[list[float], list[TelemetryLike], list[np.ndarray]], timestamp: float
) -> np.ndarray | None:
    """在 :func:`_index_telemetry` 建好的索引上做区间内插（不外推）。"""
    times, usable, poses = index
    if len(usable) < 2:
        return None
    position = bisect_left(times, timestamp)
    if position == 0 or position >= len(usable):
        return None
    lower, upper = usable[position - 1], usable[position]
    lower_pose, upper_pose = poses[position - 1], poses[position]
    span = upper.timestamp - lower.timestamp
    weight = 0.0 if span <= 0.0 else (timestamp - lower.timestamp) / span
    weight = min(1.0, max(0.0, weight))
    # body_pose_matrix 已保证四个四元数分量非空，这里逐项显式判一次，
    # 好让类型检查器也按"非空"收窄（属性访问不具备那种传递性）。
    lower_q = _quaternion_of(lower)
    upper_q = _quaternion_of(upper)
    if lower_q is None or upper_q is None:
        return None
    quaternion = _slerp(lower_q, upper_q, weight)
    matrix = np.eye(4)
    matrix[:3, :3] = quaternion_to_matrix_local(tuple(quaternion))
    matrix[:3, 3] = (1.0 - weight) * lower_pose[:3, 3] + weight * upper_pose[:3, 3]
    return matrix


def align_body_poses(
    board_poses: Sequence[tuple[float, np.ndarray, np.ndarray]],
    snapshots: Sequence[TelemetryLike],
    *,
    lag_s: float = 0.0,
    pose_at: Callable[[float], np.ndarray | None] | None = None,
) -> tuple[list[tuple[float, np.ndarray, np.ndarray]], list[np.ndarray]]:
    """把棋盘格位姿与**同一时刻**的遥测位姿配成对。

    配不上的（遥测覆盖不到的时刻）直接丢弃，返回两个等长列表
    ``(棋盘格位姿, 机体位姿)``——它们随即喂给 :func:`run_extrinsics`。

    ⚠ ``lag_s`` 的符号（这里错一次，外参就会带一个恒定的姿态偏差）：
    :func:`estimate_lag_by_correlation` 里 ``lag > 0`` 表示"**画面**比姿态滞后"，
    ``board_poses`` 使用原始 ``received_timestamp``，查询时只扣一次绝对 ``lag_s``。
    生产入口必须通过 ``pose_at`` 提供按字段源时标对齐的查询；不使用推定的
    ``capture_timestamp`` 再扣一次延迟。未提供回调的数学接口要求输入已同轴的位姿序列。
    """
    paired_board: list[tuple[float, np.ndarray, np.ndarray]] = []
    paired_body: list[np.ndarray] = []
    # 遥测索引只建一次（见 _index_telemetry 的说明：原来每个位姿都重建一遍整张表）
    index = _index_telemetry(snapshots) if pose_at is None else None
    for stamp, rvec, tvec in board_poses:
        if pose_at is None:
            assert index is not None
            pose = _telemetry_pose_from_index(index, stamp - lag_s)
        else:
            pose = pose_at(stamp - lag_s)
        if pose is None:
            continue
        paired_board.append((stamp, rvec, tvec))
        paired_body.append(pose)
    return paired_board, paired_body


def run_extrinsics(
    *,
    board_poses: Sequence[tuple[float, np.ndarray, np.ndarray]],
    body_poses: Sequence[np.ndarray] | np.ndarray,  # ndarray（逐帧堆叠）也收
    min_pair_rotation_deg: float = MIN_PAIR_ROTATION_DEG,
    min_axis_spread_deg: float = MIN_AXIS_SPREAD_DEG,
    max_pairs: int = MAX_MOTION_PAIRS,
) -> ExtrinsicsResult:
    """手眼标定：``board_poses`` 与 ``body_poses`` 必须**逐项同时刻**。

    * ``board_poses``：``(时刻, rvec, tvec)``，PnP 给出的 ``T_cam_board``；
    * ``body_poses``：机体旋转矩阵，用 :func:`align_body_poses` 按帧时刻重采样。

    ⚠ 下面两件事都要做到，否则结果没有意义：
    1. **一一对应**：遥测 10 Hz、画面 30 Hz，"第 i 条遥测配第 i 帧"是错的；
    2. **时刻一致**：扣掉画面-遥测时间差 Δt，见 :func:`align_body_poses`。

    运动对取所有 ``i < j``（超过 ``max_pairs`` 时按固定步长抽样，保证结果确定），
    ``A`` 用遥测相对运动、``B`` 用 PnP 相对运动；方程与解法见 :func:`solve_hand_eye`。

    """
    if len(board_poses) != len(body_poses):
        raise ValueError(
            f"棋盘格位姿（{len(board_poses)}）与机体位姿（{len(body_poses)}）数量不一致："
            "必须逐项同时刻，用 align_body_poses 按帧时刻重采样遥测"
        )
    if len(board_poses) < 3:
        raise ValueError(f"位姿太少（{len(board_poses)} < 3），至少需要 3 个不同位姿")

    candidates = _motion_pairs(len(board_poses), max_pairs)

    if any(np.asarray(pose).shape != (3, 3) for pose in body_poses):
        raise ValueError("机体旋转必须为 3×3 矩阵")
    body_inverse = [pose.T for pose in body_poses]
    camera_pose = [_pose_matrix(rvec, tvec) for _, rvec, tvec in board_poses]
    camera_inverse = [np.linalg.inv(item) for item in camera_pose]

    a_list: list[np.ndarray] = []
    b_list: list[np.ndarray] = []
    for index, other in candidates:
        a_matrix = body_inverse[other] @ body_poses[index]
        b_matrix = camera_pose[other] @ camera_inverse[index]
        if rotation_angle_deg(a_matrix[:3, :3]) < min_pair_rotation_deg:
            continue
        a_list.append(a_matrix)
        b_list.append(b_matrix)

    if len(a_list) < 3:
        raise ValueError(
            f"有效运动对不足（{len(a_list)} < 3）：旋转激励不够。采集时要让飞机绕**多根轴**"
            "都转起来（俯仰/横滚/偏航），全程只平移是解不出来的"
        )
    rotation, _, relative_sv, axis_spread = solve_hand_eye(a_list, b_list, use_translation=False)
    if axis_spread < min_axis_spread_deg:
        raise ValueError(
            f"相对旋转轴过于单一（最大张角 {axis_spread:.1f}° < {min_axis_spread_deg}°）："
            "所有运动对都绕同一根轴时解不唯一（OpenCV 文档同样要求至少 2 个不平行的"
            "旋转轴），请重新采集"
        )

    residuals = [
        rotation_angle_deg((a_matrix[:3, :3] @ rotation).T @ (rotation @ b_matrix[:3, :3]))
        for a_matrix, b_matrix in zip(a_list[:50], b_list[:50], strict=True)
    ]
    mean_residual = float(np.mean(residuals)) if residuals else float("nan")
    return ExtrinsicsResult(
        r_bc=rotation,
        pairs=len(a_list),
        residual_rotation_deg=mean_residual,
        axis_spread_deg=axis_spread,
        methods={"horaud": mean_residual, "relative_singular": relative_sv},
    )


def estimate_translation_fixed_rotation(
    board_poses: Sequence[tuple[float, np.ndarray, np.ndarray]],
    body_poses: Sequence[np.ndarray],
    r_bc: np.ndarray,
    *,
    max_pairs: int = MAX_MOTION_PAIRS,
) -> tuple[np.ndarray, float, int]:
    """固定已标定旋转，以同时刻的棋盘格与机体位姿求平移供人工对比。"""
    if len(board_poses) != len(body_poses) or len(board_poses) < 3:
        raise ValueError("translation_samples_insufficient")
    if any(np.asarray(pose).shape != (4, 4) for pose in body_poses):
        raise ValueError("机体位姿必须为 4×4 矩阵")
    candidates = _motion_pairs(len(board_poses), max_pairs)
    body_inverse = [np.linalg.inv(pose) for pose in body_poses]
    camera_pose = [_pose_matrix(rvec, tvec) for _, rvec, tvec in board_poses]
    camera_inverse = [np.linalg.inv(pose) for pose in camera_pose]
    relative_body: list[np.ndarray] = []
    relative_camera: list[np.ndarray] = []
    for i, j in candidates:
        body_relative = body_inverse[j] @ body_poses[i]
        relative_body.append(body_relative)
        relative_camera.append(camera_pose[j] @ camera_inverse[i])
    if not relative_body:
        raise ValueError("translation_samples_insufficient")
    matrix = np.vstack([item[:3, :3] - np.eye(3) for item in relative_body])
    rhs = np.concatenate(
        [
            r_bc @ camera[:3, 3] - body[:3, 3]
            for body, camera in zip(relative_body, relative_camera, strict=True)
        ]
    )
    if not np.isfinite(matrix).all() or not np.isfinite(rhs).all():
        raise ValueError("translation_input_non_finite")
    translation, _, rank, _ = np.linalg.lstsq(matrix, rhs, rcond=None)
    if rank < 3:
        raise ValueError("translation_geometry_rank_deficient")
    if not np.isfinite(translation).all():
        raise ValueError("translation_solution_non_finite")
    residual = float(np.sqrt(np.mean((matrix @ translation - rhs) ** 2)))
    return translation, residual, len(relative_body)


def _pose_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    import cv2  # 惰性导入：重依赖，只在真正做标定时加载（见模块 docstring）

    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return matrix


def _pose_matrix_from_rotation(rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return matrix
