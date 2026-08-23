# 自进化多语言情感分析智能体：科研实验代码设计

## 1. 目标与范围

本项目实现一个仅用于论文实验的模块化单进程框架，用于验证反馈驱动经验演化、跨语言经验迁移和动态推理策略选择。系统只调用 OpenAI-compatible API，不实现本地 Hugging Face 推理、前端、Web API 或生产部署。

首要目标是实验可复现、基线公平、组件可消融以及严格防止测试集标签泄漏。训练集以固定顺序模拟连续任务：模型先预测当前训练样本，随后获得真实标签，并据此更新经验库和策略统计。开发集用于参数选择，测试集仅用于阶段性和最终评估，测试样本及标签均不得写入经验库。

## 2. 总体架构

采用模块化单进程架构。所有组件通过 Python 接口组合，由实验运行器读取 YAML 配置后实例化。该方案避免微服务带来的额外复杂度，同时允许独立替换、测试和关闭任意研究组件。

```text
Dataset Stream
    -> SentimentAgent.predict()
        -> ExperienceRetriever
        -> StrategySelector
        -> PromptBuilder
        -> LLMClient
        -> Prediction Parser
    -> Evaluator
    -> SentimentAgent.learn(feedback)
        -> Reflector
        -> ExperienceStore
        -> StrategySelector.update()
```

预测与学习必须是两个显式阶段。`predict()` 不接收真实标签，`learn()` 只能由训练流调用。评估流仅调用 `predict()`，从接口层面降低标签泄漏风险。

## 3. 目录结构

```text
sentiment_agent/
├── configs/
│   ├── models/
│   ├── experiments/
│   └── default.yaml
├── src/sentiment_agent/
│   ├── config.py
│   ├── schemas.py
│   ├── data/
│   │   ├── loader.py
│   │   └── stream.py
│   ├── llm/
│   │   ├── client.py
│   │   ├── cache.py
│   │   └── parsing.py
│   ├── memory/
│   │   ├── store.py
│   │   ├── retrieval.py
│   │   └── scoring.py
│   ├── reflection/
│   │   └── reflector.py
│   ├── transfer/
│   │   └── cross_lingual.py
│   ├── strategies/
│   │   ├── base.py
│   │   ├── direct.py
│   │   ├── translation.py
│   │   ├── memory_augmented.py
│   │   ├── reflection_verified.py
│   │   └── selector.py
│   ├── agent/
│   │   ├── prompt_builder.py
│   │   └── agent.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── evaluator.py
│   └── experiments/
│       ├── runner.py
│       ├── baseline.py
│       ├── evolution.py
│       ├── transfer.py
│       └── ablation.py
├── scripts/
│   ├── run_experiment.py
│   └── summarize_results.py
├── tests/
├── outputs/
└── pyproject.toml
```

## 4. 核心数据模型

使用 Pydantic 定义边界数据，避免不同模块隐式传递字典。

`SentimentExample` 包含 `id`、`text`、`language`、`source`、`split` 和可选 `label`。传入 `predict()` 前必须转换为不含标签的 `PredictionInput`。

`Prediction` 包含 `label`、`confidence`、`reason`、`strategy`、`retrieved_experience_ids`、模型名、token 用量、延迟和原始响应缓存键。

`Feedback` 包含样本 ID、预测标签、真实标签、是否正确和反馈时间。它只在训练流中创建。

`Experience` 包含原始文本、语言、领域或数据源、规范化语义、真实情感、判断依据、经验类型、成功与失败计数、可靠性、创建轮次和最后使用轮次。经验类型至少区分 `successful_case`、`error_correction` 和 `generalized_rule`。

`ExperimentRecord` 保存实验 ID、配置哈希、随机种子、Git 提交、时间、模型参数、数据版本、指标和成本。

## 5. LLM 调用层

`LLMClient` 仅实现 OpenAI-compatible Chat Completions 和 Embeddings 接口。API 地址、密钥环境变量名、模型名、temperature、最大 token、超时、重试次数和并发数均由配置提供。

分类调用要求模型返回结构化 JSON：`label`、`confidence` 和 `reason`。解析器先做 JSON 提取，再做 Pydantic 校验；非法标签、空响应、限流和超时使用指数退避重试。达到重试上限后生成显式失败记录，不静默猜测标签。

为降低论文实验成本和随机性，缓存键由模型名、模型参数、系统提示和用户提示的规范化哈希组成。缓存保存完整响应、token 用量和时间。相同输入可重放，但不同实验可通过配置禁用缓存以测量真实延迟。

## 6. 经验记忆与检索

结构化经验存储使用 SQLite。向量先以序列化数组保存在数据库中，当前数据规模下由 NumPy 做余弦相似度检索，避免引入独立向量数据库。若后续规模超过单机线性检索能力，再替换为 FAISS；该替换不改变 `ExperienceRetriever` 接口。

检索器支持同语言和跨语言候选池。总分定义为语义相似度、语言匹配、领域匹配、经验可靠性和新鲜度的加权和。权重全部进入实验配置并记录。跨语言实验中，语言不匹配不再直接排除，而是降低语言项权重。

反馈更新采用确定性规则。正确经验提高成功计数与可靠性；错误经验记录错误类型并生成纠错候选。重复经验按文本哈希和语义阈值去重。互相冲突的经验不覆盖，而是分别保留统计并降低可靠性。只有高于可靠性阈值的经验能够进入 Prompt。

## 7. Reflection 与经验演化

Reflection 只在获得训练反馈后运行。输入包括脱敏后的预测输入、预测、真实标签和检索经验摘要；输出为错误原因、修正依据、可复用规则和适用范围。Reflection 结果同样使用结构化 JSON 校验。

为区分“保存正确标签”和“反思生成规则”的效果，Reflection 必须可通过配置关闭。关闭后仅保存案例型经验；开启后额外保存纠错或泛化规则。这两个条件构成关键消融实验。

## 8. 推理策略

初始实现四种策略：直接分类、翻译增强、经验增强和反思校验。所有策略实现统一的 `InferenceStrategy.build_context()` 接口，不直接调用模型。

动态选择器第一版采用按语言维护的 epsilon-greedy 多臂老虎机。奖励默认为预测正确得 1、错误得 0；可选成本惩罚用于分析准确率与 token 成本的权衡。开发集选择 epsilon，不使用测试集调参。

固定策略、随机策略和动态策略共用同一个预测管线，只替换选择器，保证比较公平。策略更新仅发生在训练反馈之后。

## 9. 实验协议

### 9.1 基线实验

实现 zero-shot、固定 prompt、few-shot、静态 RAG、基础 memory 和完整方法。所有基线使用相同基础模型、temperature、标签定义、测试集和重试策略。

### 9.2 经验演化实验

训练集顺序由固定种子决定，并按阶段切分。每个样本执行“预测后反馈”，阶段结束后在固定评估子集或完整测试集上评估，但不更新记忆。记录经验数量、Macro-F1、Accuracy、每类 F1、token 成本和延迟，绘制学习曲线。

### 9.3 跨语言迁移实验

对每个目标语言分别运行不使用经验、仅目标语言经验和加入源语言经验三种条件。输出五语言源—目标迁移矩阵。检索分数中保留领域项，以降低语言与领域混杂。

### 9.4 策略与消融实验

比较固定策略、随机策略和动态策略。消融至少包括无 Reflection、无跨语言检索、无错误经验、无可靠性过滤、无动态策略和仅原始案例。

## 10. 指标与统计

Macro-F1 为主要指标，Accuracy 和每类别 Precision、Recall、F1 为辅助指标。成本指标包括总调用数、输入输出 token、估算费用、平均延迟和失败率。

每个主要实验至少使用三个随机种子。报告均值和标准差；完整方法与最强基线在相同测试样本上进行配对比较。所有逐样本预测保存为 JSONL，以便执行 bootstrap 置信区间或配对显著性检验。

## 11. 配置、输出与复现

YAML 配置继承默认配置并覆盖实验差异。程序启动时解析为强类型配置，计算配置哈希，并将最终展开配置复制到输出目录。

Python 环境与依赖统一由 `uv` 管理。项目必须提交 `pyproject.toml` 和 `uv.lock`；开发环境使用 `uv sync --extra dev`，复现实验与 CI 使用 `uv sync --frozen --extra dev`，所有 Python、测试和 CLI 命令通过 `uv run` 执行。项目不得使用 `pip install`、Poetry、Conda 环境变更或手工维护的虚拟环境作为正式开发流程。

每次运行创建独立目录，保存 `config.yaml`、`manifest.json`、`predictions.jsonl`、`metrics.json`、`costs.json`、运行日志、经验库快照和错误记录。运行中断后可基于检查点继续，已完成样本不得重复写入反馈。

API Key 只从环境变量读取，日志和输出不得保存密钥或完整认证头。

## 12. 异常处理

API 限流、网络失败和服务端错误进行有限次数重试。解析失败单独计数。单个样本失败不终止整个实验，但必须产生带原因的失败记录。超过全局失败率阈值时主动终止，避免在服务异常时继续消耗费用。

SQLite 更新使用事务。预测成功但反馈写入失败时，不更新策略；恢复后从检查点重放该样本的反馈步骤。

## 13. 测试策略

单元测试覆盖数据拆分与标签移除、配置解析、结构化响应解析、经验去重与可靠性更新、检索排序、策略奖励更新和指标计算。API 使用假客户端，不在默认测试中发起网络请求。

集成测试使用少量合成多语言样本完成一次预测—反馈—评估闭环，并断言测试标签未进入数据库。回归测试固定假模型响应，验证不同消融配置只改变目标模块。

## 14. 实现阶段

第一阶段完成项目骨架、配置、数据模型、数据流、API 客户端、响应缓存、指标与 zero-shot 基线。第二阶段完成 SQLite 经验库、同语言检索、流式反馈和经验演化实验。第三阶段完成 Reflection、跨语言检索和迁移矩阵。第四阶段完成动态策略、全套消融、统计汇总和论文图表数据导出。

每个阶段必须先通过单元测试和小规模 dry-run，再允许产生正式实验结果。
