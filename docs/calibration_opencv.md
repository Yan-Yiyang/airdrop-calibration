# 标定与 OpenCV 5.0 易错点（`airdrop_calibration/calibrate.py`）

本文档说明：三步标定流水线（内参 → 画面/遥测时间差 → 手眼外参）的接口与数学条件，以及围绕
OpenCV 5.0 的全部易错点与判据。修改标定逻辑、改 `georef` 外参、或怀疑"某个 cv2 函数没了"时请先阅读本文档。

---

- **`airdrop_calibration/calibrate.py` = 三步标定流水线**（内参 → 画面/遥测时间差 →
  手眼外参）。代码与合成数据的回归契约见 `tests/test_calibrate.py`；它覆盖整个
  `calibrate()` 输出，包括 `t_bc` 使用尺量值以及独立的 `meta.translation_diagnostic`。
  合成回归之外，同一份 96 帧完整输入还与 eager 参考结果逐项比较了内参、畸变、旋转、延迟和平移；这一离线对照不替代实机采集验证。
  - **完整采集与内存**：默认逐帧扫描完整索引，首遍只缓存亚像素角点、时间戳、尺寸与像素摘要；内参使用所有有效视图联合求解，第二遍只重读有效视图做相位和 PnP。完整性检查、首遍和第二遍均不保留图像集合。两遍间解码像素发生变化会明确失败。`--max-frames N` 仅用于截取索引前缀调试。
  - **OpenCV API先检查实际安装包**：绑定缺失、输入类型不符和模块变更是不同问题。使用前检查`hasattr(cv2, "calibrateHandEye")`并执行最小几何调用，不能根据问题关闭或版本号推断当前wheel已可用。
    某些OpenCV 5.0 Python构建缺失手眼绑定，历史原因见官方[issue #29565](https://github.com/opencv/opencv/issues/29565)。本工具保留自研`solve_hand_eye`及同一`A X = X B`约定；替换后端需要独立数值回归，不能靠安装contrib包代替API验证。
  - **手眼的方程与来源（唯一正确写法）**：棋盘格固定在 `C = T_world_board`，
    `G = T_world_body`（**遥测**）、`P = T_cam_board`（**PnP**）、`X = T_body_cam` 待求。
    由 `G·X·P = C` 得 `A_ij = G_j⁻¹G_i`、`B_ij = P_jP_i⁻¹`，方程 **`A X = X B`**
    （与 `cv::calibrateHandEye` 同一约定）。**`A` 必须来自独立的遥测位姿**。
  - **零空间符号**：`vec(R)` 与 `vec(−R)` 表示同一旋转，但 `det(−R) = −1`。
    正交化前必须按行列式统一符号，否则可能得到差 180° 的旋转；对应处理由
    `solve_hand_eye` 的回归用例覆盖。
  - **PnP 侧三个易错点（都会给出"重投影残差很小但位姿差 180°"的结果）**：
    - `render_board` 的 `to_board` 必须减一个 `tile`：棋盘格第一个**内角点**在模板像素
      `(tile, tile)` 处，不是 `(0,0)`。少了这个偏移，渲染出的棋盘格整体错**一整格**，
      检测角点落在"板子真实点 +1 格"处，可能造成位姿错误而重投影残差仍然较小；
      用 `pattern_phase_ok` 检查棋盘格相位。
    - `solve_board_poses` 仅接受内角点列数与行数异奇偶的棋盘格；反序会翻转内部格心相位，
      以左上外格为亮格的约定可用单帧黑白对比唯一选向。同奇偶尺寸无法由该相位选向，函数明确报错。
      低对比、有效内部格样本不足或正反序候选不能唯一分开时，该帧返回 `None`。
    - `_base_look_down_pose` 的图像 `v` 轴必须与板子 `+y` **同向**：反过来合成图成了镜像，
      角点标签整体反序。另外 `look_at_board` 的扰动必须
      **旋转矩阵右乘**，不能加在旋转向量上（`|rvec|` 接近 π 时"向量相加"会剧烈跳变）。
  - **`calibrateCamera` 只收 float32——这不是 5.0 的变化**：objectPoints/imagePoints
    任一为 float64 都抛 `objectPoints should contain vector of vectors of points of type
    Point3f`；查 4.x 源码（PR #13803 的 diff，`modules/calib3d/src/calibration.cpp`）里
    写的是同样的 `checkVector(3, CV_32F)`，所以 4.x 也收不了 float64。
    真正的问题出在**我们自己**的代码：`detect_corners` 的 SB 路径返回 float64，
    `run_intrinsics` 必须显式转 float32，否则内参这步直接失败
    （`solvePnP` 则 float32/float64 都收；`moments` 同属此类既有约定，只收 int32/int64/float32）。
  - **地面标定的平移对表与主旋转/延迟隔离**：主手眼只用四元数旋转，位置异常不会改变 `R_bc` 或 `telemetry_lag`。默认开启的可选平移诊断固定主流程求得的旋转，并按 `(R_a − I)t_x = R_bc t_b − t_a` 堆叠最小二乘；估计值、尺量值、逐轴差、距离差、残差和配对数供人工判断。顶层 `t_bc` 始终写入尺量值；缺少位置或方程秩不足时报告跳过原因。位置诊断不应用可信度判据。CLI 可用 `--no-estimate-translation` 关闭。
  - **配对不能按索引**：遥测 10 Hz、画面 30 Hz，"第 i 条遥测配第 i 帧"是错的。
    用 `align_body_poses` 按帧时刻重采样；地面流程只对四元数做 slerp，平移诊断才内插位置。
    `lag_s` 的符号：`estimate_lag_by_correlation` 的 `lag > 0` 表示"画面滞后"，
    即该帧内容对应的真实时刻比帧的时刻 **早** `lag` 秒 ⇒ 查遥测用 `时刻 − lag_s`
    （`test_align_body_poses_applies_lag_with_the_right_sign`
    把它钉住了：`lag·ω` 是固定偏差，符号错了外参就带恒定姿态偏）。
  - **时间差必须用"收到帧的时刻"（`received_timestamp`）测，不能用 `capture_timestamp`**：
    索引里两条时间戳都在，`capture_timestamp = received_timestamp − 录制时的 telemetry_lag`
    是录制时**按配置假设**出来的拍摄时刻。拿它去互相关，测出来的是"该假设的残差"——
    **改动录制时的配置会平移这个残差**。所以第 2/3 步统一走 received 轴，标出的
    `telemetry_lag` = **绝对对齐量** `d_video − d_quaternion_reference`（遥测自身的滞后必须一起抵掉：
    运行时在四元数接收参照的 `收到帧时刻 − telemetry_lag` 查询源对齐遥测），可直接回填 `VideoConfig`。
    `meta.time_offset` 里另外给 `lag_baseline_s`（录制时的 lag）、
    `lag_residual_on_capture_axis_s`（capture 轴残差）与 `cross_check_ok`
    （`基准 + 残差 ≈ 绝对量`；只告警不改结果）。
    老素材缺 `received_timestamp` 时 `load_flight_stamps` **显式报错**，不静默退回 capture 轴。
    回归用例：`test_lag_measurement_is_absolute_on_received_axis`、
    `test_flight_index_time_bases_and_missing_received`。
  - **激励要求**：至少要 2 个**不平行**的旋转轴（OpenCV 文档同款要求），
    退化检查阈值 `MIN_AXIS_SPREAD_DEG = 15°`；所有运动对绕同一根轴时解不唯一，
    `run_extrinsics` 会显式报错而不是给个解。采不到旋转就重采，别调阈值。
  - `pattern_phase_ok` 发现"整体错格"这类**重投影误差不敏感**的几何错误；符号对比度
    与有效内部格数量都参与方向候选筛选，反相候选不会仅凭绝对对比度通过。
  - **仍未做**：真实素材上的复测（目前只有合成链路验证）；`lag_s` 的实测值需要真实标定采集。
  - **棋盘格渲染**：必须"深底 + 亮格"标准极性，且**模板四周不留边距**——留边距会让
    背景与边格同色、棋盘边界不可辨，`findChessboardCorners` 直接检不出
    （而 `checkChessboard` 仍返回 True，很容易误判成"几何没问题"）。
  - **`cv2.Rodrigues` 的约定遇到过两次问题**：它收的是世界→相机（相机轴作**行**）；
    与其手推旋转矩阵，不如"构造已知投影 + `solvePnP` 反解"更可靠。

## 源时标参照

合并快照的接收/组装时刻混合了异步字段，属于已确认 bug。生产流程以独立四元数源时标去重，用 `quaternion_received_at` 计算第 2 步姿态时间序列，第 3 步按同一接收参照映射到源时刻分别内插位置和姿态。输出 `meta.time_offset.telemetry_reference=quaternion_received_at` 与 `pose_alignment=source_timestamp_us`。帧始终使用原始 `received_timestamp`，只扣一次绝对标定量；原 `capture_timestamp` 用于残差互核。缺少源时标、跨重启或超过在线插值门限的断流时须重新采集连续片段。该有效对齐量包含四元数链路参照，不是独立测得的绝对视频单向延迟。
