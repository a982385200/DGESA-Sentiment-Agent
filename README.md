# ASEAN Self-Evolving Sentiment Agent

面向越南语、泰语、印尼语、马来语和高棉语低资源场景的科研实验框架。系统通过 OpenAI-compatible API 完成情感分类，并在不更新基础模型参数的前提下研究三类能力：反馈驱动的经验演化、跨语言经验迁移和动态推理策略选择。

## 研究边界

- 只实现论文实验，不包含前端或生产服务。
- 基础模型只通过 OpenAI-compatible Chat Completions 与 Embeddings API 调用。
- `predict()` 不接触真实标签；只有训练流能够在预测后创建反馈并调用 `learn()`。
- 测试集只评估，不写入经验库。
- Macro-F1 为主指标，同时保存 Accuracy、分类别指标、Token、成本、延迟和失败记录。

## 环境

项目统一使用 `uv`，Python 版本为 3.12：

```powershell
uv sync --extra dev
```

复现实验或 CI 应使用锁定依赖：

```powershell
uv sync --frozen --extra dev
```

不要使用 `pip install`、Poetry 或修改 Conda 环境。

## API 配置

复制 `.env.example` 中的变量到当前终端环境。密钥不会从配置文件读取，也不会写入实验产物。

```powershell
$env:OPENAI_API_KEY='your-key'
```

若使用其他兼容服务，在 YAML 中修改 `model.base_url`、`model.name`、`model.embedding_model` 和 `model.api_key_env`。

## 数据格式

每行是一个 JSON 对象：

```json
{"id":"vi-1","text":"dịch vụ tốt","label":"positive","language":"vi","source":"UIT-VSFC"}
```

标签必须为 `negative`、`neutral` 或 `positive`；语言代码必须为 `vi`、`th`、`id`、`ms` 或 `km`。在实验 YAML 的 `experiment.train_paths` 和 `experiment.test_paths` 中填写数据路径。

## 运行实验

先验证配置；此命令不调用 API：

```powershell
uv run python -m sentiment_agent.cli validate-config --config configs/experiments/evolution.yaml
```

运行实验：

```powershell
uv run python -m sentiment_agent.cli run --config configs/experiments/evolution.yaml
```

查看已保存的结果：

```powershell
uv run python -m sentiment_agent.cli summarize --output-dir outputs/<experiment-id>
```

`configs/experiments/` 提供 baseline、evolution、transfer、strategy 和 ablation 五类模板。正式比较必须固定基础模型、temperature、最大输出长度、重试策略、随机种子和数据划分。

## 实验产物

每次运行创建独立目录，其中包括：

- `config.yaml`：完整展开的配置；
- `manifest.json`：配置哈希、随机种子、模型、Git 提交和运行状态；
- `predictions.jsonl`：逐样本预测及策略、检索经验、标签和检查点；
- `metrics.json`：各阶段性能；
- `costs.json`：Token、估算成本和延迟；
- `experience.sqlite3`：训练反馈生成的结构化经验；
- `responses.sqlite3`：真实 API 响应缓存。

## 测试

所有默认测试完全离线，不读取真实密钥：

```powershell
uv run pytest -q
```

运行覆盖率门槛和完整流程测试：

```powershell
uv run pytest --cov=sentiment_agent --cov-report=term-missing --cov-fail-under=85
uv run pytest tests/integration/test_full_workflow.py -v
```

完整流程测试会真实执行数据加载、预测、训练反馈、Reflection、经验检索、策略更新、阶段评估和产物保存，并断言测试文本没有进入经验库。
