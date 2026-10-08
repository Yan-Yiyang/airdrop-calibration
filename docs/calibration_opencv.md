# 标定与 OpenCV 5.0 易错点（`tools/calibrate.py`）

本文档说明：三步标定流水线（内参 → 画面/遥测时间差 → 手眼外参）的实测结论，以及围绕
OpenCV 5.0 的全部易错点与判据。修改标定逻辑、改 `georef` 外参、或怀疑"某个 cv2 函数没了"时请先阅读本文档。
**内容逐字搬自 `AGENTS.md` 的"工程缺口"（P6 下半那一节），与 AGENTS.md 同等权威。**

---

- **`tools/calibrate.py`（P6 下半）= 三步标定流水线，已落地**（内参 → 画面/遥测时间差 →
  手眼外参）。采集方式：**棋盘格固定、手持飞机在其上方平移+旋转**（"板子不动、飞机在动"）。
  35 个用例全部通过（`tests/test_calibrate.py`），含**整个 `calibrate()` 跑通**的输出契约用例：
  手写一个标定飞行目录（渲染棋盘格帧 + 与真值外参自洽的遥测，遥测首末各多写一条把帧区间
  包住——`telemetry_pose_at` 拒绝外推）→ 断言 JSON 里 `t_bc` 恒为尺量值、估计值与差值进
  `meta.lever_arm`。实测这条链路：内参 RMS 0.055 px（fx 901.1 / 真值 900）、
  Δt≈0.15 s（= 素材里注入的视频链路延时；第 2 步走 received 收到时刻，给的是绝对对齐量）、
  430 个运动对、旋转残差 0.215°，杆臂估计 `[0.0584,-0.0315,0.0836]`（真值
  `[0.06,-0.03,0.09]`，差 3 mm）。
  它用的临时目录是工作区内的 `.calibrate-test-tmp`（**不用 `tmp_path`**，理由见 `AGENTS.md` 的"验证方式"）。
  - **OpenCV 相关问题的三类分因（不要一概归因于 `CV_EXPORTS_W`）**：
    **(a) 绑定标记丢失**——症状**只有一种**："`cv2` 里没有这个属性"。全库扫过 5.0 之后
    带 `bug` + `category: python bindings` 的 issue，只有两个手眼函数属于此类（见下）；
    **(b) 5.0 的行为/位置变更**——函数在，但类型、模块或后端拆分变了：FFmpeg 后端拆成
    插件 DLL（见 [`video_hm30_ffmpeg.md`](video_hm30_ffmpeg.md)）、`VideoCapture::get()` 不支持时返回 −1、DNN 换引擎；
    **(c) 一直沿用的既有约定、被我们误记成"5.0 变了"**——如 `calibrateCamera` 只收 float32、
    `moments` 不收 float64。**(b)(c) 两类都不会表现为"属性消失"，别混为一谈。**
  - **`cv2.calibrateHandEye` 在 Python 侧不可用，但原因不是"被移除"**：OpenCV 5.0 把
    `calib3d` 拆成 `geometry`/`calib`/`stereo`/`ptcloud`，函数搬进 `modules/calib`，
    **C++ 声明与签名一字未改**，但绑定标记从 `CV_EXPORTS_W` 变成了 `CV_EXPORTS`
    （实测 5.0.0 的 `calib.hpp`；同文件里 `calibrateCamera` 等仍带 `_W`），于是 Python
    侧没生成。旁证：`cv2/__init__.pyi`（343 KB 权威桩）里 `CALIB_HAND_EYE_*` /
    `HandEyeCalibrationMethod` / `RobotWorldHandEyeCalibrationMethod` 都在，唯独没有函数。
    `opencv-contrib-python` **解决不了**（contrib 用同一份主仓源码，一样没有 `_W`），
    装了还会扰动 torch/ultralytics——**不要安装**。第三步是自己实现的 `solve_hand_eye`。
    - **这是 OpenCV 官方承认的 bug，已修但还没进 wheel**（2026-09 查证）：
      报告 [issue #29565](https://github.com/opencv/opencv/issues/29565)
      "calibrateHandEye is missing in python package for version 5.0.0.93"
      （**报告的正是我们装的这个版本**，标 `bug` + `category: python bindings`，milestone 5.1，已 closed）；
      修复 [PR #29584](https://github.com/opencv/opencv/pull/29584)（merged 2026-07-25，milestone 5.1）
      只做一件事：把这两个函数从 `CV_EXPORTS` 改回 `CV_EXPORTS_W`，并补 Python 回归测试；
      4.x 的测试回移见 [PR #29693](https://github.com/opencv/opencv/pull/29693)。
      实测 `5.x` 分支的 `calib.hpp` 现在已是 `CV_EXPORTS_W void calibrateHandEye(...)`。
      **但 opencv-python 最新的 wheel 是 4.14.0.94（4.x 线，本就没丢）与 5.0.0.93（我们装的），
      5.1 尚未发 wheel**——所以在 5.1 的 wheel 落地前，自研 `solve_hand_eye` 是必需的。
      5.1 出来之后可以用 `cv2.calibrateHandEye` 反过来**交叉验证**我们的解法（注意它是
      `A X = X B` 同款约定，且 `calibrateRobotWorldHandEye` 需要 `(3,1)` 列向量）。
      **5.1 什么时候发（2026-09 查证，无官方日期）**：milestone 5.1 没有 due date
      （160 open / 223 closed），GitHub 上也没有 5.1 的任何 tag；OpenCV 历来不公布带日期的
      路线图（论坛版主原话：wiki 规划页 2024-11 后没再动）。最接近官方口径的是技术委员会
      会议纪要 `opencv/opencv/wiki/2026`——**2026-08-26 那期还在讨论"5.1 alpha?"**
      （Windows-ARM 二进制），说明当时仍处于早期阶段。按发布节奏推算：4.x 每 ~6 个月一发
      且偏爱年末（4.11.0 2025-01-09、4.13.0 2025-12-31），5.0.0 是 2026-06-06，
      **下一个 5.x minor 大约落在 2026-12–2027-01；opencv-python 的 wheel 再滞后
      1–4 周**（5.0.0 tag → 5.0.0.93 wheel 是 25 天）。**这是推算不是承诺**，
      而且 OpenCV 历史上推迟是常态。升级后一行就能判：`hasattr(cv2, "calibrateHandEye")`。
  - **手眼的方程与来源（唯一正确写法）**：棋盘格固定在 `C = T_world_board`，
    `G = T_world_body`（**遥测**）、`P = T_cam_board`（**PnP**）、`X = T_body_cam` 待求。
    由 `G·X·P = C` 得 `A_ij = G_j⁻¹G_i`、`B_ij = P_jP_i⁻¹`，方程 **`A X = X B`**
    （与 `cv::calibrateHandEye` 同一约定）。**`A` 必须来自独立的遥测位姿**。
  - **两个必须记住的数值事实**：
    1. **构造方向**：`G = P⁻¹X⁻¹` 时 `max|AX−XB| = 6.7e-16`、解到机器精度；
       写成 `G = P⁻¹X` 则 `max|AX−XB| = 0.95`（两种构造互为反向方程，极易搞混）。
    2. **零空间符号**：`vec(R)` 与 `vec(−R)` 是同一旋转，但 `det(−R) = −1`。
       若不做"按行列式整体取反"就送入"翻转最小奇异值"式正交化，会得到
       `R·diag(−1,−1,1)`——**精确差 180°**，而且**时对时错**（取决于 SVD 符号）。
       症状：零空间确实一维、真值确实在里面（`sv[-2:] = [4.17, 0.0]`），解出来却是 180°。
       回归用例跑了 5 组种子正是为了钉住这个"50% 概率"的 bug。
  - **PnP 侧三个易错点（都会给出"重投影残差很小但位姿差 180°"的结果）**：
    - `render_board` 的 `to_board` 必须减一个 `tile`：棋盘格第一个**内角点**在模板像素
      `(tile, tile)` 处，不是 `(0,0)`。少了这个偏移，渲染出的棋盘格整体错**一整格**，
      检测角点落在"板子真实点 +1 格"处（实测正面视图误差 248 px、姿态精确差 180°，
      而 PnP 残差仍只有 0.19 px）。用 `pattern_phase_ok` 把关（正确渲染 1.000）。
    - **棋盘格的标签二义性无解**：绕板面法向转 180° 后图像**完全一样**，两种"哪个角点是
      0 号"的分配都能拟合到亚像素，**方格相位也都是 1.000**（实测）——单帧图像里没有
      任何信息能定支。好在两支位姿只差一个**常量右乘**，`B'_ij = P'_jP'_i⁻¹ = B_ij`，
      **只要全局同一支，`AX=XB` 的解完全相同；混用才会失败**（残差几十度）。
      `solve_board_poses` 用**相邻帧连续性**（与上一帧夹角最小）定支，不要退回直接使用 `solvePnP`。
    - `_base_look_down_pose` 的图像 `v` 轴必须与板子 `+y` **同向**：反过来合成图成了镜像，
      角点标签整体反序（实测 4 个视图全差 180°）。另外 `look_at_board` 的扰动必须
      **旋转矩阵右乘**，不能加在旋转向量上（`|rvec|` 接近 π 时"向量相加"会剧烈跳变）。
  - **`calibrateCamera` 只收 float32——这不是 5.0 的变化**：实测 objectPoints/imagePoints
    任一为 float64 都抛 `objectPoints should contain vector of vectors of points of type
    Point3f`；查 4.x 源码（PR #13803 的 diff，`modules/calib3d/src/calibration.cpp`）里
    写的是同样的 `checkVector(3, CV_32F)`，所以 4.x 也收不了 float64。
    真正的问题出在**我们自己**的代码：`detect_corners` 的 SB 路径返回 float64，
    `run_intrinsics` 必须显式转 float32，否则内参这步直接失败
    （`solvePnP` 则 float32/float64 都收；`moments` 同属此类既有约定，只收 int32/int64/float32）。
  - **杆臂 `t_bc` 以尺量为准，标定估计值只用于对比校验**：
    `(R_a − I)t_x = R_x t_b − t_a` 需要 `t_a`（遥测位置），室内手持没有 GPS，不可信。
    所以标定文件里的 `t_bc` **恒**为文件顶部的尺量常量 `MEASURED_T_BC`（georef 读的就是
    它），估计值（`estimate_body_translation=True`，默认开）连同"与尺量值差多少"一起写进
    `meta.extrinsics.lever_arm` 供人核对——两者应当同量级、逐轴接近。
    - **判据有两条**（`t_bc_reliable`）：
      1. **残差自洽**：平移方程残差 `residual_translation_m` ≤
         `max(0.02 m, 0.05 × 画面平移跨度)`（`lever_arm_tolerance`）。`t_a` 是真是假**都**
         解得出一个 `t_x`（最小二乘照样给数），假 `t_a` 常常表现为**方程自身不自洽**。
         实测（位置与姿态**独立**抽样、尺度 0.3 m 的几何）：位置真 → 残差 `1e-15`；
         位置置零 → 残差 **1.03–2.34 m**、判为不可信（5/5）。
         ⚠ **但残差不是充分判据**：残差是右端项在 `A = stack(R_a − I)` 列空间**正交补**上的
         投影，而"遥测位置恒零/几乎不动"带来的扰动 `Δt_a` 可能**整体落在列空间里**。
         实测（本机合成采集几何：板子固定、相机原地转，即 `look_at_board` 的默认行为、
         也是项目端到端合成标定飞行的几何——它满足 `p_k = c + R_k v`）：置零前后
         **逐对残差最大差 2e-17 m**（一位不变），`t_x` 却偏 `|v| = 0.664 m`，判据照样说"可信"。
      2. **量纲合格**（`telemetry_span_ok`）：遥测位置跨度 ≥ `BODY_SPAN_FLOOR_M = 0.01 m`，
         且 `body_span_m / camera_span_m ∈ [0.1, 10]`（两边本该同量级；画面跨度为 0 =
         没有平移可比对，也判不合格）。回归用例
         `test_extrinsics_flags_zeroed_telemetry_even_though_residual_is_clean` 与
         `test_extrinsics_flags_lever_arm_when_telemetry_barely_moves` 钉住。
         它是**勤勉性检查、不是证明**：只能抓"位置恒零/几乎不动"与"两边差一个数量级"，
         验证不了"位置偏了一半"这种中间态；采不到可信位置就重采，或者只用手眼旋转解。
      **旋转解不受任何影响**（实测误差 `1.2e-6°`，因为 `A` 的旋转块与位置无关）。
      ⚠ 画面跨度在"真值位姿"下可能**恰好为 0**（相机原地转）——gate 会判不合格，这是刻意的保守。
    - **平移分支必须是堆叠最小二乘**（2026-10 修）：`(R_a − I)t_x = R_x t_b − t_a` 现在是
      **堆叠**成 `(3n×3)` 后 `lstsq`（等价于法方程 `AᵀA t = Aᵀb`）。曾经写成"把各对方程
      逐行相加再解 3×3"，那等于给每对**等权**：旋转角小的对几乎不提供信息、却把噪声全额
      带进平均。实测同一批数据上逐行相加的杆臂误差中位大 **1.4–3.5 倍**；而堆叠解与 cv2
      `calibrateHandEye` 的 **PARK** 方法在**同一个旋转解**下**逐位相同**（差 `3.4e-16 m`）。
      回归用例 `test_translation_branch_is_stacked_least_squares` 钉住（无噪声数据下两种写法
      都给真值、分辨不出，用例里专门加了确定性 PnP 扰动）。
  - **手眼解已用旧版 OpenCV 官方实现交叉验证过**（2026-10，一次性验证、脚本不进仓库）：
    隔离装 `opencv-python 4.10.0.84` / `4.14.0.94`（5.0 的 Python 侧没有 `calibrateHandEye`），
    把**同一批绝对位姿**喂给 `cv2.calibrateHandEye` 的 5 种方法与 `solve_hand_eye`：
    * 无噪声数据：两边都解到真值，最大旋转偏差 ≤ `2.1e-06°`、最大平移偏差 ≤ `7.1e-16 m`；
      把 `(G, P)` 对调后误差稳定变成 **90.109°**（说明对拍对约定敏感，不是"怎么喂都对"）；
    * 全链路（渲染 → JPEG → 检测 → 内参 → PnP）：本项目旋转误差中位 **0.075°**，
      与 cv2 最好的 PARK/HORAUD **同级**；TSAI `0.084°`、DANIILIDIS `0.085°`；
    * 角点噪声 σ=1.0 px：`1.416°` vs cv2 `1.417°`，同级；位姿数 20–30 个就够（`0.3°`）；
    * ⚠ **cv2 的 ANDREFF 在正常口径数据上 0 行有解**（`(-7:Iterations do not converge)
      Rotation normaliz…`，只在无噪声 `exact` 与部分 `attitude_noise` 行成功）——自研解没有
      这个失效模式；退化激励（全绕单轴）下本项目**显式报错**，而官方 TSAI/HORAUD 会给出
      ~90° 的错误解、DANIILIDIS 平移差 0.32 m（都不报错）；
    * 两个 4.x 版本结果一致到 `2.4e-11°`。本机产物：`.handeye-verify/`（已 gitignore：
      报告 `REPORT.md`、三阶段脚本、诊断探针、独立复核脚本）。
    - `estimate_body_translation=False` 时 `t_bc=None`、`t_bc_reliable=False`（老行为保留）。
    - **`lever_arm_report` 的 `measured` 默认值必须是 `None` 而不是 `= MEASURED_T_BC`**：
      后者是**定义时绑定**，用户改了文件顶部常量后，payload 里的 `t_bc`（运行时读取）会变、
      报告里的 `authoritative` 却是旧值，自相矛盾。
  - **配对不能按索引**：遥测 10 Hz、画面 30 Hz，"第 i 条遥测配第 i 帧"是错的。
    用 `align_body_poses` 按帧时刻重采样（位置线性内插、姿态 slerp，范围内才内插）。
    `lag_s` 的符号：`estimate_lag_by_correlation` 的 `lag > 0` 表示"画面滞后"，
    即该帧内容对应的真实时刻比帧的时刻 **早** `lag` 秒 ⇒ 查遥测用 `时刻 − lag_s`
    （`test_align_body_poses_applies_lag_with_the_right_sign`
    把它钉住了：`lag·ω` 是固定偏差，符号错了外参就带恒定姿态偏）。
  - **时间差必须用"收到帧的时刻"（`received_timestamp`）测，不能用 `capture_timestamp`**（2026-10 修）：
    索引里两条时间戳都在，`capture_timestamp = received_timestamp − 录制时的 telemetry_lag`
    是录制时**按配置假设**出来的拍摄时刻。拿它去互相关，测出来的是"该假设的残差"——
    **换一次配置就平移一个 lag**（真机上表现为"把上次标定值回填、再录一次，Δt 自己收敛到 0"，
    但那不代表标定对了）。受控实验（固定 `d_video`/`d_tel`、只改录制时的 lag）：

    | 真实 d_video | 录制 lag_cfg | 遥测 d_tel | 用 capture 轴测 | 用 received 轴测 |
    | --- | --- | --- | --- | --- |
    | 0.15 | 0.15 | 0.00 | 0.0000 | **0.1495** |
    | 0.15 | 0.15 | 0.05 | −0.0499 | **0.0997** |
    | 0.30 | 0.15 | 0.05 | 0.0997 | **0.2493** |
    | 0.15 | 0.00 | 0.00 | 0.1495 | 0.1495 |

    每行两条轴之差恰好等于 `lag_cfg`。所以第 2/3 步统一走 received 轴，标出的
    `telemetry_lag` = **绝对对齐量** `d_video − d_tel`（遥测自身的滞后必须一起抵掉：
    运行时按 `拍摄时刻 = 收到时刻 − telemetry_lag` 取遥测），可直接回填 `VideoConfig`。
    `meta.time_offset` 里另外给 `lag_baseline_s`（录制时的 lag）、
    `lag_residual_on_capture_axis_s`（capture 轴残差）与 `cross_check_ok`
    （`基准 + 残差 ≈ 绝对量`，实测差 ≤0.5 ms；只告警不改结果）。
    老素材缺 `received_timestamp` 时 `load_flight_stamps` **显式报错**，不静默退回 capture 轴。
    回归用例：`test_lag_measurement_is_absolute_on_received_axis`、
    `test_flight_index_time_bases_and_missing_received`，以及端到端契约用例里
    "注入视频延时 0.15 ⇒ 标出 ≈0.15（而非 ≈0）"这条断言。
  - **激励要求**：至少要 2 个**不平行**的旋转轴（OpenCV 文档同款要求），
    实测阈值 `MIN_AXIS_SPREAD_DEG = 15°`；所有运动对绕同一根轴时解不唯一，
    `run_extrinsics` 会显式报错而不是给个解。采不到旋转就重采，别调阈值。
  - `pattern_phase_ok` 的用途只有一个：发现"整体错格"这类**重投影误差不敏感**的几何错误；
    它**不能**消解标签二义性（见上）。
  - **仍未做**：真实素材上的复测（本机只有合成链路 + 内参/时间差的合成验证）；
    `lag_s` 的实测值需要一次真实标定采集。（手眼解本身已用旧版 OpenCV 官方实现交叉验证，
    见上一条。）
  - **棋盘格渲染**：必须"深底 + 亮格"标准极性，且**模板四周不留边距**——留边距会让
    背景与边格同色、棋盘边界不可辨，`findChessboardCorners` 直接检不出
    （而 `checkChessboard` 仍返回 True，很容易误判成"几何没问题"）。
  - **`cv2.Rodrigues` 的约定遇到过两次问题**：它收的是世界→相机（相机轴作**行**）；
    与其手推旋转矩阵，不如"构造已知投影 + `solvePnP` 反解"更可靠。
  - **不要再试**（都试过、都失败）：靠方格相位区分正序/反序标签；把 `A` 写成
    "棋盘格相对机体"的相对运动（那等于用 `X` 自身构造 `A`，`A ≡ B`，真值不在解空间里）；
    用 C 序 reshape 拆零空间向量（得到的是另一个"看着也像旋转矩阵"的解）。
