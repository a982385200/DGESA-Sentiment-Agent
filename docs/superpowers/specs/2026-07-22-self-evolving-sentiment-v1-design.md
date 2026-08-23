# 自进化多语言情感分析智能体 V1 设计

## 1. 目标

第一版实现一个面向论文实验的最小完整闭环：系统使用固定基础 Prompt，通过 LangChain 调用 Qwen API 完成三分类情感预测；使用本地多语言 Embedding 检索历史经验；训练流在 mini-batch 预测结束后揭示真实标签，并按输入顺序更新经验；下一批预测能够使用前一批产生的经验。

本版本验证的核心研究命题是：在不更新基础模型参数的情况下，反馈驱动的结构化经验积累是否能改善后续情感分类结果。

## 2. 范围

V1 包含：

- 越南语、泰语、印尼语、马来语和高棉语三分类数据加载；
- LangChain Qwen OpenAI-compatible API 后端；
- 本地 BGE-M3 Embedding 后端；
- 固定基础 Prompt 和动态经验上下文；
- SQLite 结构化经验存储；
- NumPy 精确向量检索；
- mini-batch 预测、反馈和有序学习；
- 成功案例与错误纠正案例两类经验；
- Accuracy、Macro-F1、各类别 Precision/Recall/F1；
- API 响应缓存、调用用量、延迟和失败记录；
- 配置、数据指纹、预测、指标和经验库等实验产物；
- 完全离线的单元测试和端到端集成测试。

V1 不包含：

- 自动重写全局基础 Prompt；
- LLM 深度错误归因和泛化规则生成；
- 经验关系图、复杂冲突消解及经验删除；
- 显式跨语言迁移权重学习；
- 动态推理策略选择；
- FAISS 或外部向量数据库；
- Web 服务或前端界面。

这些能力在 V1 闭环验证后分阶段加入，避免多个机制同时变化而无法解释实验结果。

## 3. 研究语义

### 3.1 固定 Prompt 与可进化经验

基础 Prompt 固定定义任务、标签、判断原则和 JSON 输出格式。实验过程中不修改基础 Prompt。每次预测的最终上下文由以下部分组成：

```text
固定基础 Prompt
+ 当前输入文本和语言
+ 检索到的结构化经验
+ JSON 输出约束
```

因此，V1 的进化对象是外部经验，而不是基础模型参数或全局 Prompt。

### 3.2 训练与评估隔离

训练样本先转换成不含真实标签的 `PredictionInput`。预测阶段不得访问真实标签。整批预测成功后，实验运行器才能用真实标签构造 `Feedback` 并调用学习阶段。

开发集和测试集仅用于评估，只调用预测接口，不写入经验库，也不更新任何学习状态。实验产物必须能够证明测试样本没有进入经验库。

### 3.3 Mini-batch 进化

每个训练批次采用以下确定性语义：

1. 在批次开始时读取经验快照；
2. 一次性对本批文本执行本地向量化；
3. 按输入顺序完成经验检索和 Prompt 构造；
4. 在受限并发下调用 Qwen 完成预测；
5. 无论 API 完成顺序如何，都按输入顺序组装结果；
6. 整批预测成功后揭示真实标签；
7. 按输入顺序创建反馈并更新经验；
8. 本批产生的经验从下一批开始可见。

`batch_size=1` 必须与严格逐样本在线学习等价。评估检查点是硬边界，批次不得跨越配置的检查点。

若本批任一预测最终失败，则本批不执行学习，避免产生部分更新。失败被记录并由运行配置决定重试或终止。

## 4. 总体架构

```text
DatasetLoader
    -> ExperimentRunner
        -> SentimentAgent.predict_batch()
            -> LocalEmbeddingBackend
            -> ExperienceRetriever
            -> PromptBuilder
            -> LangChainLLMBackend
        -> FeedbackFactory
        -> SentimentAgent.learn_batch()
            -> ExperienceUpdater
            -> ExperienceRepository
        -> Evaluator
        -> ArtifactWriter
```

模块使用显式接口组合。实验运行器负责流程，Agent 负责预测与学习协调，模型后端只负责模型调用，经验模块只负责经验生命周期。评估器不得持有可写经验仓库。

## 5. 主要模块

### 5.1 配置

配置使用 YAML 加 Pydantic 强类型校验。配置至少覆盖：

- 数据文件和语言；
- Qwen 模型名、Base URL、API Key 环境变量名；
- temperature、最大输出 token、超时、重试及并发；
- 本地 Embedding 模型路径或 ModelScope 模型 ID、设备和 batch size；
- 经验检索数量、最低可靠性和检索评分权重；
- 训练 batch size、检查点、随机种子和输出目录；
- 是否启用经验、缓存及跨语言候选。

API Key 只从环境变量读取，不允许写入 YAML、日志或实验产物。每次运行保存完全展开且已脱敏的配置及配置哈希。

### 5.2 数据层

统一数据边界包含 `id`、`text`、`label`、`language` 和 `source`。标签只允许 `positive`、`neutral`、`negative`，语言只允许 `vi`、`th`、`id`、`ms`、`km`。

加载时校验空文本、重复 ID、非法标签和非法语言。为每个输入文件生成内容哈希，并记录在运行清单中。训练顺序由显式随机种子决定，开发集与测试集保持固定。

### 5.3 LLM 后端

`LLMBackend` 定义与 LangChain 解耦的批量分类接口。V1 实现 `LangChainQwenBackend`，内部使用 `langchain-openai` 的 `ChatOpenAI` 连接 Qwen OpenAI-compatible API。

模型输出必须通过 Pydantic 校验为：

```json
{
  "label": "positive | neutral | negative",
  "confidence": 0.0,
  "reason": "判断依据"
}
```

若原生结构化输出不可用，则使用 JSON Prompt、本地提取和校验。限流、超时、服务端错误和解析错误执行有限重试；不得在失败时静默猜测标签。

缓存键由模型名、模型参数和规范化消息内容的哈希构成。缓存保存响应、token 用量和延迟，不保存 API Key。

### 5.4 本地 Embedding

`EmbeddingBackend` 定义 `embed(texts)` 批量接口。V1 使用 ModelScope 或本地缓存加载 BGE-M3。输出向量转为 `float32` 并执行 L2 归一化。

单元测试使用确定性的 Fake Embedding，不下载模型。真实模型加载测试归入显式 online/model 测试，不在默认测试中执行。

### 5.5 经验模型

V1 经验字段包括：

- `id`；
- `type`：`successful_case` 或 `error_correction`；
- `language`；
- `source`；
- `text`；
- `semantic_summary`；
- `sentiment`；
- `reason`；
- `reliability`；
- `success_count`；
- `failure_count`；
- `source_sample_id`；
- `created_batch`；
- `last_used_batch`；
- `status`：V1 固定使用 `active`。

正确预测生成 `successful_case`，错误预测生成 `error_correction`。V1 不额外调用 LLM 生成泛化规则；`semantic_summary` 和 `reason` 使用预测理由与真实标签构造的可审计内容。后续版本再增加深度归因。

### 5.6 经验存储与向量索引

SQLite 保存经验、使用结果和事件历史。至少包含：

- `experiences`：经验当前状态；
- `experience_outcomes`：经验在某次预测中的检索和结果；
- `experience_events`：创建、强化和惩罚事件；
- `schema_metadata`：数据库模式版本。

经验向量保存在运行目录中的 NumPy 矩阵，向量 ID 映射和 Embedding 模型元数据分别保存。SQLite 更新使用事务；向量索引通过临时文件写入后原子替换，避免半写状态。

每个实验使用独立可写经验库。需要相同初始经验时，从同一只读快照初始化。

### 5.7 检索

SQLite 先筛选 `active` 且可靠性达到阈值的候选经验，NumPy 对归一化向量执行精确余弦相似度检索。最终评分由以下可配置项组成：

```text
score = semantic_similarity
      + language_match_weight
      + source_match_weight
      + reliability_weight
```

所有评分组成随预测记录保存。V1 可以允许跨语言候选，但不学习迁移权重，也不把这一行为作为跨语言创新的最终实现。

### 5.8 V1 经验更新

V1 使用确定性更新规则：

- 当前预测正确：创建或合并成功案例；被注入且与结果一致的经验成功计数增加；
- 当前预测错误：创建或合并错误纠正案例；被注入且支持错误预测的经验失败计数增加；
- 被检索但未注入 Prompt 的经验不更新；
- 无法判断贡献的经验不盲目强化。

经验可靠性采用 Beta-Bernoulli 平滑：

```text
reliability = (success_count + 1) /
              (success_count + failure_count + 2)
```

去重键由规范化文本、语言、数据源、真实情感和经验类型组成。每次状态变化追加事件，不覆盖历史。

### 5.9 实验运行器

实验运行器负责：

- 固定种子和训练顺序；
- mini-batch 分组和并发控制；
- 检查点硬边界；
- 预测后反馈；
- 周期性只读评估；
- 运行清单、日志、指标、预测和成本输出；
- 完成批次后的检查点；
- 从检查点恢复且不重复学习已完成样本。

V1 至少支持 zero-shot 和 self-evolving 两种条件。两者共享相同模型、基础 Prompt、数据顺序、温度和评估流程；唯一差异是经验检索与更新是否启用。

## 6. 项目结构

代码按已确认的顶层边界组织：`data`、`llm`、`embeddings`、`prompts`、`experience`、`agent`、`evaluation`、`experiments` 和 `reporting`。V1 只创建当前闭环需要的文件，保留接口以便后续加入归因、迁移和策略模块，不预先创建无实现的空模块。

运行产物结构为：

```text
outputs/<experiment_id>/
├─ resolved_config.yaml
├─ manifest.json
├─ run.log
├─ predictions.jsonl
├─ metrics.json
├─ costs.json
├─ checkpoints/
└─ experience_store/
   ├─ experiences.sqlite3
   ├─ vectors.npy
   ├─ vector_ids.json
   └─ metadata.json
```

## 7. CLI 与易用性

V1 提供以下命令：

```text
sentiment-agent validate-config
sentiment-agent run
sentiment-agent summarize
sentiment-agent experience list
sentiment-agent experience show
sentiment-agent experience stats
sentiment-agent experience history
sentiment-agent experience export
```

所有 Python 和 CLI 命令通过 `uv run` 执行。README 提供环境准备、`.env` 配置、离线测试、单次实验、批量实验和经验查看说明。SQLite 不要求独立服务或数据库账号。

## 8. 错误处理

- 配置、数据和模型输出错误尽早失败并给出明确上下文；
- API 临时错误有限重试，达到上限后记录原始异常类型；
- mini-batch 预测失败时不执行该批学习；
- SQLite 事务失败时不更新向量索引；
- 向量矩阵、ID 映射和数据库记录不一致时拒绝继续运行；
- 日志不得包含 API Key、完整认证头或未脱敏环境变量；
- 默认禁止覆盖已有实验目录。

## 9. 测试策略

开发遵循测试驱动：每个公共函数或方法先添加失败的独立单元测试，再实现生产代码。

单元测试覆盖：

- 配置解析与秘密信息脱敏；
- 数据校验、标签移除和数据指纹；
- Qwen 后端消息映射与结构化解析；
- 本地 Embedding 批量接口；
- 固定 Prompt 与经验注入；
- SQLite CRUD、事务和事件历史；
- 向量索引一致性与检索排序；
- 经验去重、强化、惩罚和可靠性；
- mini-batch 结果顺序、快照可见性和失败原子性；
- 评估指标与测试集无泄漏；
- CLI 参数和产物生成。

完整离线集成测试使用 Fake LLM、Fake Embedding 和极小多语言数据，真实执行：

```text
加载数据
→ 第一批预测
→ 反馈和经验写入
→ 第二批检索新增经验
→ 只读测试评估
→ 保存并检查全部产物
```

默认测试不读取真实 `.env`，不下载模型，不调用外部 API。Qwen 连接和 BGE-M3 加载使用显式标记的手动测试。

## 10. V1 验收条件

V1 完成必须同时满足：

1. `uv sync --frozen --extra dev` 可建立环境；
2. 完整默认测试离线通过；
3. 所有公共函数和方法有独立单元测试；
4. 完整离线集成闭环通过；
5. `batch_size=1` 与逐样本在线语义一致；
6. 同批经验不可见，下一批经验可见；
7. 测试样本和标签不会写入经验库；
8. SQLite 可通过 CLI 查询并导出 CSV/JSONL；
9. 使用 `.env` 中 Qwen 配置可完成一个显式的小规模在线 smoke run；
10. 输出包含复现实验所需的配置哈希、数据指纹、随机种子、Git 提交和模型信息；
11. README 能让新用户仅依靠 `uv` 完成配置、离线测试和实验运行。

## 11. 后续演进

V1 通过后，第二版加入错误样本的 LLM 深度归因、正确样本的确定性贡献归因、泛化规则、冲突关系和候选经验晋升。随后分别加入显式跨语言迁移机制与上下文策略学习，保证每个创新点都能单独消融并得到清晰证据。
