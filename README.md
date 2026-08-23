# ASEAN Self-Evolving Sentiment Agent

面向越南语、泰语、印尼语、马来语和高棉语的经验驱动自进化情感分析研究框架。

项目使用本地多语言 Embedding 和 LangChain Qwen API，在不更新基础模型参数的情况下研究 mini-batch 反馈驱动的经验演化。Python 环境和依赖统一由 `uv` 管理。

默认从项目内的 `models/embeddings/bge-m3` 加载 Embedding，并强制使用本地文件。Zero-shot 配置关闭检索，因此不会加载或计算真实 Embedding。

## 环境

Python 3.12 环境与依赖只使用 `uv`：

```powershell
uv sync --frozen --extra dev
```

复制 `.env.example` 为 `.env`，填写 `OPENAI_API_KEY`（这里存放 Qwen 兼容 API Key）。密钥不会写入配置、日志或实验产物。SQLite 由 Python 自带，不需要安装数据库服务、账号或端口。

## 测试

默认测试完全离线，不读取 `.env`、不下载 BGE-M3、也不调用 Qwen：

```powershell
uv run python -m pytest -q
uv run python -m pytest tests/integration/test_offline_evolution_workflow.py -v
```

## 运行

先执行不调用 API 的配置校验：

```powershell
uv run sentiment-agent validate-config --config configs/experiments/evolution.yaml
```

运行自进化实验：

```powershell
uv run sentiment-agent run --config configs/experiments/evolution.yaml
```

运行时默认显示阶段、样本/批次进度、成功失败数、token、经验数量、耗时和 ETA。批处理脚本可关闭动态显示：

```powershell
uv run sentiment-agent run --config configs/experiments/evolution.yaml --no-progress
```

配置中的 `train_batch_size` 控制 mini-batch 大小。同一批共享批次开始时的经验快照，整批预测成功后再按原顺序学习；`train_batch_size: 1` 等价于逐样本在线学习。

## 查看经验

```powershell
uv run sentiment-agent experience stats --run outputs/<experiment-id>
uv run sentiment-agent experience list --run outputs/<experiment-id>
uv run sentiment-agent experience show --run outputs/<experiment-id> --id <experience-id>
uv run sentiment-agent experience history --run outputs/<experiment-id> --id <experience-id>
uv run sentiment-agent experience export --run outputs/<experiment-id> --format csv
```

也可以选择安装 DB Browser for SQLite，直接打开 `experience_store/experiences.sqlite3`；图形工具不是项目依赖。

每次运行保存配置、数据指纹、模型信息、逐样本预测、指标、调用用量、经验数据库和向量索引。测试集只执行预测与评估，不写入经验库。
