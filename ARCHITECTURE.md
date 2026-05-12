# 月面压力闭环 VLA 演示系统架构

本文档面向人类读者，用于解释论文用真机演示系统的目标、系统边界、运行拓扑、闭环状态机和仍需确认的实验参数。

## 1. 系统目标

本项目从空白架构重新开始，不继续修补旧的 `lunar_maintenance_demo`。目标是构建一个论文用真机演示系统：机器人通过相机观察机械压力表，Gemini 只负责读表，本地确定性逻辑计算压力误差并选择规范化 VLA prompt，OpenPI/RoboCOIN 负责执行短时阀门操作。

已确定范围：

- 只支持一个目标管路、一个压力表、一个阀门。
- 不保留旧 repo 的 simulator、mock VLA、HTTP VLA、ROS stub、通用 CLI。
- 新 repo 以 Python 脚本作为入口，不再以 `lunar-demo ...` CLI 作为主要操作方式。
- Gemini 只做视觉读表，不生成机器人动作，不做高层规划。
- 本地 planner 必须是确定性的，负责从压力误差生成阀门动作意图。
- OpenPI/VLA 只接收短时、固定措辞的 prompt。
- RoboCOIN 机器人侧尽量保持黑盒式工作方式：持续采集 observation，发送给 policy server，接收 action chunk 并执行。

明确不做：

- 不做多阀门、多压力表、多管线扩展。
- 不做非压力任务透明转发；非本任务直接跑 OpenPI server，不经过本 repo。
- 不接入独立电子压力传感器。压力来源是相机图像中的机械压力表读数。
- 不依赖力、力矩、卡死、打滑等遥测信号，因为当前真机拿不到这些信号。
- 不做阀门角度估计或机械限位视觉估计。
- 不在本 repo 中训练 VLA，不直接控制机器人关节。

## 2. 风险判断原则

任何“需要确认”的接口、协议、动作语义、模型能力、硬件行为，都不能靠猜测写死。

编码代理或实现者在判断风险点前，必须先查阅相关源码、配置、运行命令、官方文档或最小实验结果，并在实现说明中写明依据。如果调研结论仍不确定，文档和代码中必须保留显式 TODO/保护分支，不能把不确定项伪装成已确认事实。

对机器人动作安全相关内容采用最保守策略：宁可不发送动作，也不要发送语义不清的默认动作。

## 3. 系统边界

### 3.1 LunarPressure 新 repo

负责：

- 运行 OpenPI-compatible orchestration server。
- 接收 RoboCOIN `robot_client_openpi.py` 发来的 observation。
- 从 observation 中取出相机图像，主要是 `observation.scene_image`，调用 Gemini 读压力表。
- 校验 Gemini JSON、置信度和视觉读数。
- 计算目标压力与当前压力的误差。
- 生成规范化 OpenPI prompt。
- 在一个动作阶段内，多次把规范化 prompt 和最新 observation 转发给真实 OpenPI policy server。
- 记录图像、Gemini 原始响应、解析结果、planner 决策、规范化 prompt、OpenPI action chunk 和 timing。
- 提供最小 dashboard 或 replay API，用于查看最近一次实验。

不负责：

- 机器人 SDK 初始化。
- 机器人动作下发。
- OpenPI 模型训练。
- 非压力任务。

RoboCOIN 常用机器人侧命令：

```bash
uv run src/lerobot/scripts/server/robot_client_openpi.py \
  --host="192.168.1.166" \
  --port=8001 \
  --task="<端到端任务描述，例如 `Keep Line-A pressure at 0.1 MPa`>" \
  --robot.type=realman \
  --robot.ip="192.168.1.17" \
  --robot.port=8080 \
  --robot.block=False \
  --robot.init_type="none" \
  --robot.use_canfd=True \
  --robot.canfd_follow=False \
  --robot.wait_second=0.0 \
  --robot.joint_cmd_threshold_deg=0.05 \
  --robot.velocity=100 \
  --robot.cameras="{ observation.scene_image: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, observation.wrist_image: {type: intelrealsense, serial_number_or_name: \"243322073824\", width: 640, height: 480, fps: 30}}" \
  --camera_keys="[ observation.scene_image, observation.wrist_image ]" \
  --robot.id=rm75_follower \
  --frequency=30 \
  --timing_log_interval=5 \
  --timing_log_window=10
```

LunarPressure 必须兼容该命令发出的图像 key 和 observation key。

### 3.2 RoboCOIN

负责：

- 机器人连接、相机采集、关节状态读取。
- 按固定频率调用 policy server。
- 接收 `action` 或 `actions` chunk 并逐步执行。
- 当前 demo 不 patch RoboCOIN stop hook；STOP 时由 LunarPressure 返回经验证的 hold action，RoboCOIN 仍走原有 `send_action(action)` 路径保持当前姿态。

当前已知 observation 行为：

- `robot_client_openpi.py` 会发送 `prompt`，其值来自 `--task`。
- 它会构造 `observation.state`。
- 它还会构造 `observation.joint_position`、`observation.joint_velocity`、`observation.gripper_position`。
- 对所有 `observation.*` key，它会额外生成 slash 版本，例如 `observation.scene_image` -> `observation/scene_image`。
- 当前实际相机 key 倾向使用 `observation.scene_image` 和 `observation.wrist_image`。

### 3.3 openpi-lunarbot

负责：

- 运行真实 OpenPI policy server。
- 加载 RM75 微调后的 checkpoint。
- 接收规范化 prompt 和 observation。
- 返回 action chunk。

当前 RM75 policy 侧支持的相关输入：

- `observation.scene_image` / `observation.wrist_image`
- `observation.images.scene_image` / `observation.images.wrist_image`
- `observation.state`
- 拆分后的关节字段：`observation.joint_position`、`observation.joint_velocity`、`observation.gripper_position`
- `prompt`

本 repo 转发给 OpenPI 时保持 RoboCOIN 原始 observation，只替换 `prompt`。真机前仍需用 smoke test 确认当前 openpi-lunarbot checkpoint 接受该原始 key 集合。

## 4. 运行拓扑

推荐真机运行形态：

```text
机器人 PC / RM75 侧
  RoboCOIN robot_client_openpi.py
    发送类似 `Keep Line-A pressure at 0.1 MPa` 的高层 task
    采集相机图像和机器人状态
    将 observation 发送到 ws://<gpu-ip>:<lunarpressure-port>
    接收 action chunk 或 hold action
    执行机器人动作

GPU 工作站
  LunarPressure orchestration server
    对 RoboCOIN 暴露 OpenPI websocket 协议
    调用 Gemini API 读取压力表
    将高层任务改写为规范化 VLA prompt
    将规范化 prompt 转发给真实 OpenPI server

  openpi-lunarbot policy server
    提供微调后的 RM75 OpenPI policy
    接收规范化 prompt
    返回 action chunk
```

重要原则：RoboCOIN 始终发送原始高层 `--task`，例如 `Keep Line-A pressure at 0.1 MPa`。LunarPressure 负责维护闭环状态，并决定每个阶段应该发送哪个规范化 prompt 给 OpenPI。

## 5. 闭环状态机

系统不应该把一次 `infer()` 调用视为一次完整阀门动作。一个规范化 VLA prompt 可能需要在动作阶段中连续执行多个 action chunk，机器人动作才算有实际完成意义。

### 5.1 `OBSERVE`

输入：

- 最新 RoboCOIN observation。
- 场景相机和腕部相机图像。
- 当前高层任务。

行为：

1. 选择要发送给 Gemini 的图像。主路径优先避免磁盘往返，但 Gemini API 可能最终需要 image bytes。
2. 必要时允许 Gemini 或本地预处理做缩放、裁剪、聚焦压力表区域。
3. 调用 Gemini 压力表读数 prompt。
4. 将严格 JSON 解析为本地 `GaugeReading`。
5. 记录 Gemini 原始响应和解析后的结构化结果。
6. 如果解析失败或置信度过低，保持在 `OBSERVE` 状态，并在新帧上重试。

输出：

- 有效视觉压力读数，或停止/重试决策。

### 5.2 `PLAN`

输入：

- 目标压力。
- 容差。
- 最新 `GaugeReading`。

行为：

1. 计算压力误差：`target_pressure_mpa - current_pressure_mpa`。
2. 如果误差在容差内，转入 `STOP`。
3. 如果当前压力低于目标，选择能升高压力的阀门方向。
4. 如果当前压力高于目标，选择能降低压力的阀门方向。
5. 保守选择动作幅度。

输出：

- 一个 `LocalPlan`，不是模型生成的计划。

### 5.3 `ACT`

输入：

- planner 选择的规范化 prompt。
- websocket 循环中收到的最新 RoboCOIN observation。

行为：

1. 在动作阶段内，将每个新的 observation 携带规范化 prompt 转发给真实 OpenPI server。
2. 将 OpenPI 返回的 action chunk 回传给 RoboCOIN。
3. 保存 action chunk 和 timing 元数据。
4. 动作完成后回到 `OBSERVE`。

动作完成不应该只写死一个固定时间窗。已收口的硬兜底为每个 ACT 阶段最多 `3` 次 policy 调用或 `10` 秒，并结合 action chunk / 关节残差的收敛迹象判断是否结束。但实现前必须确认 action 语义。

风险说明：

- 不能直接假设 OpenPI 返回的是 delta action。需要根据实际 openpi-lunarbot policy config 和 RoboCOIN robot config 判断。
- 当前源码调研显示，RM75 数据管线在训练输入侧会把绝对关节目标转换为 delta，但输出侧有 `RM75AbsoluteActions` 把动作恢复为绝对关节目标；RoboCOIN client 随后直接调用 `robot.send_action(action)`。
- RoboCOIN 的 robot config 还存在 `delta_with` 选项，默认是 `none`，但实际运行命令可能改变动作解释方式。
- 因此动作完成检测不能简单看 action 本身范数。更稳的候选是：
  - 若实际发送的是绝对关节目标，检查“目标关节位置 - 当前 observation.state”的残差是否收敛；
  - 检查相邻 action chunk 的变化量是否低于阈值；
  - 设置最大 policy 调用次数、最大动作时间和人工急停兜底；
  - 通过一次小幅离线/假机器人实验确认 action 语义后再启用真机策略。

### 5.4 `STOP`

输入：

- 达到目标、触发不安全条件、操作员停止，或出现不可恢复的视觉感知失败。

行为：

1. 向 RoboCOIN 返回经验证的 hold action。
2. RoboCOIN 不应把 stop prompt 当成普通操作 prompt 再发给 VLA。
3. RoboCOIN 按原有 action 执行路径调用 `send_action(action)`，但 action 语义必须是保持当前姿态。

## 6. 规范化 VLA Prompt

规范化 prompt 是数据采集、训练和推理阶段都使用的固定字符串。这样可以减少歧义，并防止 Gemini 或任何 LLM 自由发明机器人动作。

已收口 prompt 集合：

```yaml
canonical_prompts:
  from_observation_pose:
    increase_pressure: "Move from the observation pose, grasp valve V1, and turn it clockwise slightly."
    decrease_pressure: "Move from the observation pose, grasp valve V1, and turn it counterclockwise slightly."
  already_grasping_valve:
    increase_pressure: "While holding valve V1, turn it clockwise slightly."
    decrease_pressure: "While holding valve V1, turn it counterclockwise slightly."
  stop_data_label: "Stop and hold the arm at the current pose."
```

收口结论：

- VLA 数据集里的 task label 就是普通 text，会保存在 LeRobot V2.1 格式数据集的 `metadata.json` 内，例如：

```json
{
  "task": "Pick up .... and place ......"
}
```

- 在真实物理设置中，`counterclockwise` 是降低压力。
- 在真实物理设置中，`clockwise` 是升高压力。
- prompt 中统一使用 `Line-A/G1/V1` 配置中的 `V1`，不再使用旧示例中的 `V2`。
- 论文 demo 只保留一种动作幅度：`slightly`。不保留 `tiny` 动作。
- 数据需要覆盖两类初始状态，并拆成两类 task label：
  - 机械臂完全没有接触阀门，需要从初始位置移动到阀门处，捏住阀门并拧动。
  - 机械臂已经抓住阀门，处于调整过程中，只需要继续微调。
- `stop_data_label` 可以作为数据标签保留，但闭环执行时不能作为普通 OpenPI prompt 下发；停止必须走停止协议。

## 7. Observation 契约

### 7.1 输入侧 RoboCOIN Observation

预期 key：

```text
prompt
observation.scene_image
observation.wrist_image
observation.state
observation.joint_position
observation.joint_velocity
observation.gripper_position
observation/scene_image
observation/wrist_image
observation/state
observation/joint_position
observation/joint_velocity
observation/gripper_position
```

每个组件只需要其中一部分：

- Gemini 压力表读取器需要图像数组。
- OpenPI policy server 需要图像数组、状态/本体感知信息，以及规范化 `prompt`。
- Logger 需要图像快照、prompt、action chunk 和 timing。

### 7.2 内部数据表示

对延迟敏感的主路径应优先使用内存中的 numpy array。

除非某个库强制要求，否则调用 Gemini 或 OpenPI 前不要把图像写入磁盘。日志系统可以异步保存图像，或在响应发送后再保存。

### 7.3 转发给 OpenPI 的 Observation

除非明确决定要统一 key 命名，否则 LunarPressure 应转发原始 observation，只把 `prompt` 替换为规范化 prompt。

已收口：

- Gemini 主图像使用 `observation.scene_image`。
- Gemini fallback 图像使用 `observation.wrist_image`。
- 转发给 OpenPI 时保持 RoboCOIN 发来的原始 observation，只替换 `prompt` 为规范化 prompt。
- OpenPI 路径不做图像 resize/convert。
- 压力表读数路径由 LunarPressure 负责组织，允许 Gemini API 按 prompt 做 zoom/crop/preprocess；这只影响 Gemini 读表，不改变转发给 OpenPI 的原始图像。

## 8. 停止协议

停止时不要继续把 OpenPI policy server 的动作原样转发给 RoboCOIN。

必须注意：**不能默认发送全 0 action 来停止机器人。**

当前源码调研结论：

- openpi-lunarbot 的 RM75 action transform 写明 canonical action 是 8 维：7 维绝对关节位置 + 1 维 gripper。
- RM75 训练侧使用 `RM75DeltaActions` 将绝对关节目标转换为 delta，但输出侧使用 `RM75AbsoluteActions` 将动作加回当前 state，恢复为绝对关节目标。
- RoboCOIN `robot_client_openpi.py` 从响应中取出 `action/actions` 后，逐个调用 `robot.send_action(action)`。
- RoboCOIN `BaseRobotConfig.delta_with` 默认是 `none`，也就是默认按绝对控制解释；如果命令显式设置 `delta_with=previous/initial`，动作语义会变化。

已收口的停止策略：

```yaml
stop_protocol:
  patch_robot_client_for_lunar_control_stop: false
  stop_response_mode: "hold_send_action"
  stop_behavior: "hold"
```

实现含义：

- 本次论文 demo 暂不 patch RoboCOIN client 识别 `lunar_control.stop=true`。
- STOP 时 LunarPressure 返回经验证的 hold action，让 RoboCOIN 继续调用 `send_action(action)`，但动作语义是保持当前姿态。
- hold action 必须由当前 observation 中的真实机器人状态构造，不能使用全 0 action。
- 在启用真机 STOP 前，必须先确认 action key 顺序、单位、gripper 语义、`delta_with` 配置，以及 `observation.state`/`observation.joint_position` 到 hold action 的映射。
- 必须用 dummy policy 或 fake robot 验证 hold response 不会导致 RoboCOIN 崩溃，并且 `send_action()` 接收到的是当前姿态 hold action。

## 9. 安全模型

这里的安全模型是论文 demo 层面的保守逻辑，不是经过认证的机器人安全系统。

当前可用的安全信息来源：

- Gemini JSON schema 是否有效。
- Gemini 置信度。
- 视觉压力读数硬边界。
- 最大动作时间窗或最大 policy 调用次数。
- 最大 observe/act 循环次数。
- 停止命令路径。
- 本 repo 之外的人类/操作员急停。

当前不可用：

- 独立压力传感器。
- 力/力矩遥测。
- 卡死/打滑检测。
- 阀门角度估计。

停止条件草案：

- Gemini 在 N 次重试后仍无法返回有效压力。
- Gemini 置信度持续低于阈值。
- 视觉压力读数超出硬边界。
- 达到闭环最大循环次数。
- 达到目标压力。
- 操作员请求停止。

## 10. 当前配置草案

物理设置：

```yaml
line_id: "Line-A"
gauge_id: "G1"
valve_id: "V1"
target_pressure_mpa: 0.1
tolerance_mpa: 0.05
visual_hard_min_mpa: 0.0
visual_hard_max_mpa: 0.2
pressure_unit: "MPa"
```

阀门方向约定：

```yaml
operator_camera_view_definition: "压力表和阀门的斜前方"
increase_pressure_direction: "clockwise"
decrease_pressure_direction: "counterclockwise"
notes: ""
```

规范化 prompt：

```yaml
use_tiny_action: false
canonical_prompts:
  from_observation_pose:
    increase_pressure: "Move from the observation pose, grasp valve V1, and turn it clockwise slightly."
    decrease_pressure: "Move from the observation pose, grasp valve V1, and turn it counterclockwise slightly."
  already_grasping_valve:
    increase_pressure: "While holding valve V1, turn it clockwise slightly."
    decrease_pressure: "While holding valve V1, turn it counterclockwise slightly."
  stop_data_label: "Stop and hold the arm at the current pose."
```

动作完成判据：

```yaml
act_completion:
  max_policy_calls_per_act: 3
  max_act_seconds: 10
  use_joint_residual_check: true
  use_action_delta_check: true
  joint_residual_threshold: "TODO: 通过 fake robot 或小幅真机标定后填写"
  action_delta_threshold: "TODO: 通过 fake robot 或小幅真机标定后填写"
```

停止协议：

```yaml
stop_protocol:
  patch_robot_client_for_lunar_control_stop: false
  stop_response_mode: "hold_send_action"
  stop_behavior: "hold"
```

Gemini 压力表读取器：

```yaml
model: "gemini-robotics-er-1.6-preview"
image_keys_to_send:
  - "observation.scene_image"
  - "observation.wrist_image"
confidence_threshold: 0.8
max_retries_per_observe: 5
prompt_file: ""
response_schema: "GaugeReading"
preprocess_owner: "LunarPressure"
resize_policy: "不 resize；Gemini 读表路径可由 Gemini API 做 zoom/crop/preprocess，OpenPI 路径保持原图"
```

Observation key：

```yaml
observation_keys:
  gemini_primary_image: "observation.scene_image"
  gemini_fallback_image: "observation.wrist_image"
  forward_to_openpi: "保持原始 observation，仅替换 prompt"
```

端口和主机建议全部放入 `.env`：

```yaml
lunarpressure_server_host: "192.168.1.166"
lunarpressure_server_port: 8001
openpi_backend_host: "127.0.0.1"
openpi_backend_port: 8000
robot_pc_ip: "192.168.1.124"
gpu_workstation_ip: "192.168.1.166"
```

OpenPI 后端示例命令：

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_rm75_pick_place \
    --policy.dir=/home/bingqi/data/admins/bingqi/Projects/openpi-lunarbot/checkpoints/pi05_rm75_pick_place/rm75_pick_place_lora_bs48_b_side/16000/
```

## 11. 日志与 replay

必须记录：

- 配置快照。
- 输入的高层任务。
- 输入 observation 元数据。
- 所有 policy loop 图像帧。
- Gemini prompt。
- Gemini 原始响应。
- 解析后的 `GaugeReading`。
- 本地 plan。
- 规范化 prompt。
- 转发给 OpenPI 的 observation 元数据。
- OpenPI 原始响应。
- Action chunk。
- 循环 timing。
- 停止/最终结果。

建议路径：

```text
runs/
  <run_name>/
    .log
    images/
    actions/
```

## 12. 附录：Gemini API 参考

实现者需要进一步确认 Gemini API 对图像 bytes、代码执行、缩放/裁剪和严格 JSON 输出的支持方式。当前设计允许 Gemini 或本地预处理做 zoom/crop，以提高压力表读数稳定性；但最终输出必须是严格 JSON，并经过本地 schema 校验后才能参与 planner。

用户提供的参考要点：

- Gemini Robotics ER 示例可通过 `types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg')` 传入图像。
- 可在 prompt 中要求模型 zoom/crop 图像以确认仪表指针位置。
- 可使用 system instruction 强制模型只返回 JSON。
- 如启用 code execution，必须明确记录这一事实，并把原始响应写入日志。

## 13. 操作员运行流程

标准真机启动顺序：

1. 确认物理设置：目标管路为 `Line-A`，压力表为 `G1`，阀门为 `V1`，相机视角为压力表和阀门的斜前方。
2. 确认安全边界：目标压力 `0.1 MPa`，容差 `0.05 MPa`，视觉硬边界 `[0.0, 0.2] MPa`。
3. 启动真实 openpi-lunarbot policy server，加载 RM75 微调 checkpoint。
4. 启动 LunarPressure orchestration server，对 RoboCOIN 暴露 OpenPI-compatible websocket 协议。
5. 在机器人 PC 启动 RoboCOIN `robot_client_openpi.py`，`--host/--port` 指向 LunarPressure server，`--task` 使用高层任务，例如 `Keep Line-A pressure at 0.1 MPa`。
6. LunarPressure 进入 `OBSERVE -> PLAN -> ACT -> OBSERVE` 闭环；Gemini 只读表，本地 planner 只选择固定 prompt，OpenPI 执行短时阀门动作。
7. 达到目标、视觉失败、超出安全边界、达到循环上限或操作员停止时进入 `STOP`；当前策略是返回经验证的 hold action。
8. 实验结束后检查 `runs/<run_name>/` 中的配置快照、图像、Gemini raw response、planner 决策、OpenPI response、action chunk 和 timing。

## 14. 真机前验证项

架构参数已收口，但以下项目必须在真机运行前通过源码确认或最小实验验证：

1. hold action 的 action key 顺序、单位、gripper 语义和 `delta_with` 配置。
2. `joint_residual_threshold` 和 `action_delta_threshold` 的数值。
3. fake robot/dummy policy 路径下，`hold_send_action` 不会导致 RoboCOIN 崩溃。
4. Gemini API 的图像 bytes、strict JSON、zoom/crop/preprocess 行为符合实现预期。
