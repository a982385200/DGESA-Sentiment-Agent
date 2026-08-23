# DGESA Sentiment Agent

本仓库提供“双粒度经验进化情感分析智能体”（Dual-Granularity Experience Self-Evolution Agent, DGESA）的核心实现。

DGESA 将经验划分为样本级经验和模式级经验，通过反馈驱动的经验生成、有效性检验、准入、对齐、抽象、生命周期管理及双通道检索，支持东盟小语种情感分析。

## 核心流程

1. 使用样本级经验与模式级经验辅助情感预测。
2. 根据预测结果和真实标签生成双粒度候选经验。
3. 检验候选经验的有效性，并控制样本经验准入。
4. 对模式经验进行语义对齐、抽象更新和生命周期管理。
5. 冻结经验库后，在测试集上计算分类指标。

论文对齐实现位于 `src/sentiment_agent/dgesa/`。

## 环境要求

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- OpenAI 兼容的 Qwen API
- 本地 Sentence Transformers 兼容嵌入模型

安装依赖：

```powershell
uv sync --extra dev
```

复制环境变量示例：

```powershell
Copy-Item .env.example .env
```

然后在 `.env` 中填写自己的 API Key：

```dotenv
OPENAI_API_KEY=your-api-key
```

不要将 `.env` 或真实 API Key 提交到 Git。

## 准备嵌入模型

公开配置不包含嵌入模型权重。下载兼容模型后，修改 `configs/dgesa_paper.example.yaml`：

```yaml
embedding:
  model_id: /path/to/your/embedding/model
  device: cpu
```

如需使用 NVIDIA GPU，请安装 CUDA 版 PyTorch，并把 `device` 改为 `cuda`：

```powershell
uv pip install --reinstall torch --torch-backend cu130
uv run python -c "import torch; print(torch.cuda.is_available())"
```

## 示例数据

仓库仅提供越南语 mini 数据用于接口验证：

```text
datasets/mini_dataset/vietnamese/train.json  # 100 条
datasets/mini_dataset/vietnamese/test.json   # 100 条
```

数据来源为 UIT-VSFC（Vietnamese Students' Feedback Corpus）。完整实验数据不包含在本仓库中，应按照相应数据集的许可条款单独获取。

## 配置

公开配置文件为：

```text
configs/dgesa_paper.example.yaml
```

运行前至少需要设置：

- `model.name`：调用的 Qwen 模型名称。
- `model.base_url`：OpenAI 兼容 API 地址。
- `embedding.model_id`：本地嵌入模型路径或模型标识。
- `embedding.device`：`cpu` 或 `cuda`。

验证配置结构：

```powershell
uv run sentiment-agent validate-config --config configs/dgesa_paper.example.yaml
```

## 运行

使用越南语 mini 数据执行 DGESA：

```powershell
uv run sentiment-agent run-paper --config configs/dgesa_paper.example.yaml
```

该命令会真实调用配置的 LLM API。调用成本、速率限制和数据发送策略由所使用的 API 服务决定。

## 输出

运行结果写入 `outputs/<timestamp>-paper-<config-hash>/`：

- `metrics.json`：测试集分类指标。
- `train_predictions.jsonl`：训练阶段预测。
- `test_predictions.jsonl`：测试阶段预测。
- `sample_experiences.jsonl`：准入后的样本级经验。
- `pattern_experiences.jsonl`：模式级经验及其生命周期状态。
- `experience_store/dgesa.sqlite3`：双粒度经验数据库。
- `manifest.json`：运行状态和经验规模。

`outputs/` 默认被 Git 忽略。

## 测试

默认测试不会访问外部 API，也不需要真实凭证：

```powershell
uv run pytest -q
```

运行论文算法的离线集成测试：

```powershell
uv run pytest tests/integration/test_paper_dgesa_workflow.py -v
```

## 目录结构

```text
configs/dgesa_paper.example.yaml   公开配置模板
datasets/mini_dataset/vietnamese/  越南语 mini 示例数据
src/sentiment_agent/dgesa/         DGESA 核心实现
tests/                             单元测试与离线集成测试
```

## 发布范围

本仓库面向学术审阅和代码参考，包含核心算法与最小示例，但不包含：

- 完整实验数据集
- 嵌入模型权重
- 真实实验配置和 API 凭证
- 论文源文件、图表及完整实验输出

因此，示例可以用于验证代码接口和核心流程，但不能直接复现论文中的完整实验结果。
