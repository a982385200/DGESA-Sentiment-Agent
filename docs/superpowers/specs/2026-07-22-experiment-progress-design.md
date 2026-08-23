# 实验进度显示设计

## 目标

为 `sentiment-agent run` 增加默认启用的 Rich 动态进度显示，让 zero-shot 和自进化实验在批次执行期间报告阶段、样本数、批次、成功/失败、token、耗时和 ETA；使用 `--no-progress` 可关闭显示。

## 架构

实验运行器只产生与 UI 无关的结构化 `ProgressEvent`，通过 `ProgressReporter.update(event)` 通知展示层。`RichProgressReporter` 负责终端渲染，`NullProgressReporter` 负责无输出运行。运行器默认使用 Null 实现，因此库调用和测试不依赖终端；CLI 根据 `--progress/--no-progress` 注入实现。

## 事件模型

事件包含：

- `stage`：`train`、`dev` 或 `test`；
- `completed_samples` 与 `total_samples`；
- `completed_batches` 与 `total_batches`；
- `successful_requests` 与 `failed_requests`；
- `input_tokens` 与 `output_tokens`；
- `elapsed_seconds`；
- `experience_count`；
- 可选 `checkpoint`。

事件只在一个 mini-batch 全部成功后递增完成数量。当前 V1 的批次失败会传播异常，因此失败计数在正常进度中保持零；后续引入可跳过失败样本时沿用同一字段。

## 终端行为

CLI 默认显示一个动态 Rich 进度区域。训练和评估阶段切换时重置样本与批次总数，累计 token 和整体耗时保持运行级统计。描述行展示阶段、批次、成功/失败、token 和经验数；Rich 根据已完成比例显示百分比与 ETA。

非交互终端仍允许 Rich 输出。用户可通过 `--no-progress` 完全关闭动态显示。进度展示发生异常时记录或忽略展示异常，不改变预测、学习、指标或实验产物。

## 测试

先用 `RecordingProgressReporter` 验证运行器按批次发出有序事件、完成数量单调递增、阶段正确、token 正确，以及训练后经验数更新。CLI 测试验证默认创建 Rich reporter，`--no-progress` 选择 Null reporter。Rich 渲染测试使用内存 Console，不依赖真实终端、API 或模型。

## 验收条件

- 默认运行命令显示动态进度；
- `--no-progress` 无动态输出；
- zero-shot 显示 test 阶段且不加载 Embedding；
- 自进化显示 train/dev/test 阶段和经验数；
- 每批只更新一次完成进度；
- 全套离线测试通过；
- 进度机制不改变预测顺序、经验可见性或实验指标。
