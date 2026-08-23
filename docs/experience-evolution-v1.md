# 经验进化 V1 使用说明

本版本实现论文核心链路：训练样本先保存为案例证据；错误样本由 Qwen API 做结构化归因；候选规则使用本地 BGE-M3 向量进行在线相似合并；规则获得跨 mini-batch 支持后升级为 `active`；预测阶段只检索 `active` 通用经验。它不创建或调用 Skills。

## 运行

项目根目录的 `.env` 需要包含 `OPENAI_API_KEY`。Embedding 从项目内 `models/embeddings/bge-m3` 读取，不需要联网下载。

```powershell
uv run sentiment-agent validate-config --config configs/experiments/evolution_mini.yaml
uv run sentiment-agent run --config configs/experiments/evolution_mini.yaml
```

配置中的 `experiment.train_batch_size` 控制在线学习批大小。一个批次预测时使用批次开始前的规则快照；整个批次预测结束后才统一学习，因此不会发生同批数据泄漏。

## 经验生命周期

- `candidate`：已经提取，但证据不足，不参与预测。
- `active`：支持数、支持批次数、可靠度和矛盾率满足阈值，参与预测。
- `conflicted`：已激活规则后来出现过多矛盾，暂停参与预测。
- `suppressed`：矛盾非常明显，不参与预测。

默认需要至少 2 个支持案例且来自 2 个不同 mini-batch 才能激活。阈值都在 `generalization` 配置段中，可以用于消融实验。

## 实验产物

每次运行在 `outputs/<run-id>/` 保存：

- `predictions.jsonl`：train/test 的逐样本预测、标签、理由和检索规则 ID。
- `experience_store/experiences.sqlite3`：案例、归因、规则、证据关系、生命周期事件和使用结果。
- `generalized_experiences.jsonl`：通用经验规则快照。
- `attributions.jsonl`：错误归因记录。
- `attribution_failures.jsonl`：仅在归因 JSON 解析失败并使用确定性回退时生成。
- `experience_evolution_metrics.json`：案例数、归因数、各状态规则数、支持/矛盾数和压缩率。
- `metrics.json`、`costs.json`、`manifest.json`：任务指标、API 用量和可复现元数据。

SQLite 是单文件数据库，Python 已内置支持，不需要安装数据库服务。可以用 DB Browser for SQLite 打开；重点查看 `case_evidence`、`attributions`、`generalized_experiences`、`generalized_experience_evidence`、`experience_events` 和 `experience_outcomes` 表。
