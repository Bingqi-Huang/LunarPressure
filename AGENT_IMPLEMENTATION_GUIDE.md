# Coding Agent 实现指南

本文档面向编码代理。实现前必须先阅读 `ARCHITECTURE.md`，并严格遵守本文的接口、风险和实现顺序约束。

## 1. 工作原则

- 不迁移旧 simulator 代码。
- 不保留旧 CLI。
- 除非真实 OpenPI 路径需要，否则不保留通用 VLA adapter。
- 主闭环逻辑应集中在一个 orchestrator 状态机中。
- Gemini 应作为 `GaugeReader`，不要再叫 `Brain`。
- planner 必须是确定性的，并且容易测试。
- 所有任务相关常量放在统一配置文件或 `.env` 中。
- 优化主路径延迟：调用 Gemini/OpenPI 前避免不必要的图像磁盘往返。
- 日志要完整，但不能阻塞主路径。
- 初期测试只覆盖契约和确定性逻辑。

## 2. 风险处理硬约束

任何“需要确认”的接口、协议、动作语义、模型能力、硬件行为，都不能靠猜测写死。

编码代理必须做到：

1. 先查源码、配置、运行命令、官方文档或最小实验结果。
2. 在实现说明中写明依据。
3. 如果仍不确定，保留显式 TODO/保护分支。
4. 对动作安全采用最保守策略：宁可不发送动作，也不要发送语义不清的默认动作。

特别注意：**禁止在未确认动作语义前发送全 0 action chunk 作为停止动作。**

## 3. 已确认代码事实

这些事实来自本地源码调研，后续如果相关 repo 更新，需要重新验证。

### 3.1 RoboCOIN action 流

- `robot_client_openpi.py` 调用 `response = self.policy.infer(obs)`。
- 随后从 response 提取 `action` 或 `actions`。
- 每个 action 经过 `_prepare_action()` 后直接调用 `self.robot.send_action(action)`。
- `_prepare_action()` 只处理 action 维度和 gripper channel，不判断 `lunar_control.stop`。

实现影响：

- 如果 LunarPressure 返回 `lunar_control.stop=true`，RoboCOIN 当前不会自动识别，必须 patch 或包装 robot client。本次已收口为不采用该路径，改用经验证的 `hold_send_action`。
- 如果 response 中没有可提取 action，当前 RoboCOIN 可能报错。停止协议必须先经过 dummy/fake robot 验证。

### 3.2 openpi-lunarbot RM75 action 语义

- `openpi-lunarbot/src/openpi/research/shared/action_transforms.py` 写明 canonical action 是 8 维：7 维绝对关节位置 + 1 维 gripper。
- 训练输入侧 `RM75DeltaActions` 会把绝对关节目标转换为 delta。
- 输出侧 `RM75AbsoluteActions` 会将动作加回当前 state，恢复为绝对关节目标。

实现影响：

- 不能简单假设 OpenPI 对 RoboCOIN 输出的是 delta action。
- 动作完成检测不能直接看 action 本身范数。
- 停止时不能默认发 0 action。

### 3.3 RoboCOIN delta 配置

- `BaseRobotConfig.delta_with` 默认是 `none`。
- 如果命令显式设置 `delta_with=previous` 或 `delta_with=initial`，RoboCOIN 会改变动作解释方式。

实现影响：

- 每次真机运行前必须记录实际 RoboCOIN 命令和 robot config。
- 停止/hold action 语义必须基于实际 `delta_with` 确认。

## 4. 目标 repo 结构

建议结构：

```text
configs/
  lunar_pressure.yaml
  prompts/
    gemini_gauge_reading.md

scripts/
  run_orchestrator_server.py
  run_gemini_gauge_smoke.py
  run_latest_dashboard.py

src/lunar_pressure/
  __init__.py
  config.py
  schemas.py
  observation_contract.py
  gemini_gauge_reader.py
  local_planner.py
  canonical_prompts.py
  orchestrator.py
  openpi_proxy_server.py
  openpi_backend_client.py
  logging.py
  dashboard.py

tests/
  test_observation_contract.py
  test_gemini_schema.py
  test_local_planner.py
  test_orchestrator_state_machine.py

runs/
  <run_name>/
    .log
    images/
    actions/
```

## 5. 配置契约

端口、host、API key、模型名等运行环境参数应来自 `.env` 或统一配置，不要散落在代码里。

当前配置草案：

```yaml
line_id: "Line-A"
gauge_id: "G1"
valve_id: "V1"
target_pressure_mpa: 0.1
tolerance_mpa: 0.05
visual_hard_min_mpa: 0.0
visual_hard_max_mpa: 0.2
pressure_unit: "MPa"

increase_pressure_direction: "clockwise"
decrease_pressure_direction: "counterclockwise"
operator_camera_view_definition: "压力表和阀门的斜前方"

use_tiny_action: false
canonical_prompts:
  from_observation_pose:
    increase_pressure: "Move from the observation pose, grasp valve V1, and turn it clockwise slightly."
    decrease_pressure: "Move from the observation pose, grasp valve V1, and turn it counterclockwise slightly."
  already_grasping_valve:
    increase_pressure: "While holding valve V1, turn it clockwise slightly."
    decrease_pressure: "While holding valve V1, turn it counterclockwise slightly."
  stop_data_label: "Stop and hold the arm at the current pose."

act_completion:
  max_policy_calls_per_act: 3
  max_act_seconds: 10
  use_joint_residual_check: true
  use_action_delta_check: true
  joint_residual_threshold: null  # TODO: 通过 fake robot 或小幅真机标定后填写
  action_delta_threshold: null    # TODO: 通过 fake robot 或小幅真机标定后填写

stop_protocol:
  patch_robot_client_for_lunar_control_stop: false
  stop_response_mode: "hold_send_action"
  stop_behavior: "hold"

gemini_model: "gemini-robotics-er-1.6-preview"
gemini_confidence_threshold: 0.8
gemini_max_retries_per_observe: 5
gemini_primary_image: "observation.scene_image"
gemini_fallback_image: "observation.wrist_image"
image_preprocess_owner: "LunarPressure"
resize_policy: "不 resize；Gemini 读表路径可由 Gemini API 做 zoom/crop/preprocess，OpenPI 路径保持原图"
forward_to_openpi: "保持原始 observation，仅替换 prompt"

lunarpressure_server_host: "192.168.1.166"
lunarpressure_server_port: 8001
openpi_backend_host: "127.0.0.1"
openpi_backend_port: 8000
robot_pc_ip: "192.168.1.124"
gpu_workstation_ip: "192.168.1.166"
```

必须标定后才能真机运行：

- `joint_residual_threshold`
- `action_delta_threshold`
- hold action 的 action key 顺序、单位、gripper 语义和 `delta_with` 配置

## 6. Schema 要求

建议至少定义：

```text
PressureTask
GaugeReading
LocalPlan
CanonicalPrompt
OrchestratorState
OpenPIActionResponse
LunarControl
StepLogRecord
RunSummary
```

`GaugeReading` 至少包含：

```text
line_id
gauge_id
value_mpa
confidence
raw_text
need_retry
risk_flags
```

所有 Gemini 输出必须先经过 schema 校验，再进入 planner。

## 7. Observation Contract

输入侧必须兼容 RoboCOIN 的 key：

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

实现建议：

- `observation_contract.py` 提供统一函数读取 dot/slash key。
- 不要在业务代码里到处手写 key fallback。
- 默认 Gemini 用 `observation.scene_image`，`observation.wrist_image` 可作为 fallback 或日志材料。
- 转发给 OpenPI 时优先保持原始 observation，只替换 `prompt`。

## 8. Orchestrator 状态机

状态：

```text
OBSERVE
PLAN
ACT
STOP
ERROR
```

### 8.1 OBSERVE

职责：

- 从最新 observation 提取图像。
- 调用 Gemini gauge reader。
- 校验 JSON 和 confidence。
- 失败时根据 retry policy 重试。

必须记录：

- 输入图像元数据。
- Gemini prompt。
- Gemini raw response。
- 解析后的 `GaugeReading`。

### 8.2 PLAN

职责：

- 计算 `target_pressure_mpa - current_pressure_mpa`。
- 在容差内进入 `STOP`。
- 低于目标则选择升压方向。
- 高于目标则选择降压方向。
- 生成规范化 prompt。

硬约束：

- planner 不能调用 LLM。
- planner 不能输出自由文本动作。
- planner 只能从配置中的规范化 prompt 集合选择。

### 8.3 ACT

职责：

- 在动作阶段内转发 observation 到 OpenPI backend。
- 将 OpenPI action chunk 回传给 RoboCOIN。
- 记录 action chunk 和 timing。
- 根据动作完成判据或兜底上限返回 `OBSERVE`。

风险约束：

- 实现前必须确认 action 语义。
- 不能只看 action 本身范数作为唯一完成判据。
- 兜底上限已收口为每个 ACT 阶段最多 `3` 次 policy 调用或 `10` 秒。
- `use_joint_residual_check=true` 和 `use_action_delta_check=true`，但阈值必须通过 fake robot 或小幅真机标定后才能启用真机策略。

### 8.4 STOP

职责：

- 不再转发 OpenPI 动作。
- 当前收口策略为返回经验证的 hold action，让 RoboCOIN 走原有 `send_action(action)` 路径保持当前姿态。
- 记录停止原因。

禁止：

- 未确认语义前返回全 0 action chunk。
- 未确认 action key 顺序、单位、gripper 语义和 `delta_with` 前启用 hold action 真机停止路径。

## 9. Gemini Gauge Reader

职责：

- 接收图像 bytes 或数组。
- 构造 pressure gauge reading prompt。
- 调用 Gemini。
- 可允许 zoom/crop/code execution，但必须记录原始响应。
- 输出严格 `GaugeReading`。

要求：

- `temperature` 等生成参数要集中配置。
- JSON 解析失败时不得进入 planner。
- 低于 confidence threshold 时不得进入 ACT。
- 支持最多 `max_retries_per_observe` 次重试。

## 10. OpenPI Backend Client

职责：

- 连接真实 openpi-lunarbot policy server。
- 转发 observation。
- 替换 `prompt` 为规范化 prompt。
- 接收 `action` 或 `actions`。
- 保留 `server_timing` 和 raw response。

要求：

- 不修改图像和 state；OpenPI 路径保持原始 observation，只替换 `prompt`。
- 不吞掉 backend 错误。
- 每次 backend 调用都写入 step log。

## 11. RoboCOIN Stop Strategy

当前收口策略是不 patch RoboCOIN client：

```yaml
stop_protocol:
  patch_robot_client_for_lunar_control_stop: false
  stop_response_mode: "hold_send_action"
  stop_behavior: "hold"
```

实现要求：

1. STOP 时 LunarPressure 返回一个经验证的 hold action，而不是 `lunar_control.stop=true` 空 action。
2. hold action 必须从当前 observation 中的真实机器人状态构造，目标是保持当前姿态。
3. 如果 action 语义尚未确认，STOP 必须进入 `ERROR` 或人工停止路径，不能猜测 hold action。
4. 禁止用全 0 action 代替 hold action。

验证要求：

- 用 dummy policy 或 fake robot 验证 hold response 不会使 RoboCOIN 崩溃。
- 验证 `send_action()` 收到的是当前姿态 hold action。
- 验证 stop 后视频/log 能正常保存。
- 小幅真机动作前确认 action key 顺序、单位、gripper 语义和 `delta_with`。

## 12. 日志

每个 run 建议保存：

```text
runs/
  <run_name>/
    .log
    config_snapshot.yaml
    summary.json
    steps.jsonl
    images/
    actions/
    gemini_raw/
```

每个 step 记录：

- 时间戳。
- 当前状态。
- 输入 high-level task。
- observation key 列表和 shape/dtype 元数据。
- 图像保存路径。
- Gemini prompt/raw/parsed。
- planner 输出。
- 规范化 prompt。
- OpenPI raw response。
- action chunk 保存路径。
- timing。
- stop/final outcome。

## 13. 实现阶段

建议顺序：

1. 定义 schema 和配置加载。
2. 定义 observation extraction contract。
3. 实现 Gemini gauge reader，先支持本地图片 smoke test。
4. 实现确定性 local planner 和规范化 prompt compiler。
5. 实现 OpenPI backend client。
6. 实现 OpenPI-compatible websocket orchestrator server。
7. 实现日志。
8. 实现 `hold_send_action` 停止策略，并保留未确认 action 语义时的保护分支。
9. 实现 smoke scripts。
10. 实现最小 dashboard/latest-run viewer。

## 14. 测试要求

初期测试重点：

- observation dot/slash key fallback。
- Gemini JSON schema 校验。
- local planner 压力误差到 prompt 的映射。
- counterclockwise 降压、clockwise 升压。
- STOP 不返回全 0 action。
- orchestrator 状态转移。
- OpenPI backend response 中 `action` 和 `actions` 两种字段兼容。
- 日志 schema。

真机前 smoke test：

1. Gemini 对保存图像读表。
2. Orchestrator 连接 dummy OpenPI server。
3. RoboCOIN 连接 orchestrator，但 fake robot/dummy action。
4. Stop response 验证 `send_action()` 收到当前姿态 hold action，而不是全 0 action。
5. 小幅真机动作验证 action 语义，并标定 `joint_residual_threshold` / `action_delta_threshold`。

## 15. 真机前验证项

架构参数已收口，但以下项目必须在真机运行前通过源码确认或最小实验验证：

1. hold action 的 action key 顺序、单位、gripper 语义和 `delta_with` 配置。
2. `joint_residual_threshold` 和 `action_delta_threshold` 的数值。
3. fake robot/dummy policy 路径下，`hold_send_action` 不会导致 RoboCOIN 崩溃。
4. Gemini API 的图像 bytes、strict JSON、zoom/crop/preprocess 行为符合实现预期。
