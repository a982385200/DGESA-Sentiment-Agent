# 归因驱动的通用经验进化核心设计

## 1. 目标

在现有 mini-batch 情感分析系统中实现论文的经验进化核心，使系统不再把每个训练样本直接作为最终检索经验，而是形成以下可审计闭环：

```text
案例证据
→ 错误归因
→ 候选规则
→ 相似规则合并
→ 跨批次验证
→ 通用经验晋升
→ 后续推理使用
```

基础模型参数和固定任务 Prompt 不更新。进化对象是通用经验的内容、支持证据、冲突证据、可靠性、适用范围和生命周期。

## 2. 范围

本版本实现：

- 所有训练反馈保存为案例证据；
- 错误样本使用确定性初筛和 Qwen 结构化归因；
- 归因结果形成候选通用规则；
- 使用本地 BGE-M3 对候选规则进行向量化；
- 在标签和范围兼容前提下按语义阈值合并规则；
- 根据跨批次支持数和冲突率晋升或抑制规则；
- 推理阶段优先检索 active 通用经验；
- SQLite 保存案例、归因、通用经验、证据关系和事件历史；
- 实验产物记录归因调用、规则数量、压缩率和规则使用情况；
- 通过配置关闭归因或通用化，以支持消融实验。

本版本不实现：

- Skill 或自动生成执行流程；
- 复杂知识图谱；
- 自动修改基础 Prompt；
- HDBSCAN 等参数复杂的聚类方法；
- 前端和人工审核界面；
- 自动删除历史证据；
- 完整领域识别模型。

## 3. 核心实体

### 3.1 CaseEvidence

每个训练样本在反馈揭示后保存一条不可变案例证据：

```text
id
sample_id
text
language
source
predicted_label
gold_label
prediction_reason
confidence
retrieved_experience_ids
batch_id
correct
created_at
```

案例证据只用于审计、归因、规则支持和冲突分析，不直接作为默认 Prompt 经验。

### 3.2 Attribution

错误样本保存结构化归因：

```text
id
case_id
error_type
root_cause
corrected_reason
candidate_rule
scope_languages
scope_sources
phenomena
confidence
raw_response
status
created_batch
```

`error_type` 固定为：

```text
missing_knowledge
wrong_experience
retrieval_failure
negative_transfer
negation_error
domain_knowledge_error
reasoning_error
label_ambiguity
```

### 3.3 GeneralizedExperience

最终可检索的通用经验：

```text
id
status: candidate | active | conflicted | suppressed
semantic
sentiment
rule
corrected_reason
error_types
scope_languages
scope_sources
phenomena
support_count
contradiction_count
supporting_batches
reliability
created_batch
last_updated_batch
version
```

通用经验向量由 `semantic + rule + corrected_reason` 生成，保存在独立 NumPy 索引中。

## 4. 数据库

在现有 SQLite 中新增：

```text
case_evidence
attributions
generalized_experiences
generalized_experience_evidence
generalized_experience_events
generalized_experience_outcomes
```

现有 `experiences`、`experience_events` 和 `experience_outcomes` 保留，用于旧实验兼容；新实验默认不再把案例写入旧经验检索索引。

`generalized_experience_evidence` 记录：

```text
experience_id
case_id
attribution_id
relation: support | contradiction
batch_id
```

同一个案例对同一规则只能产生一次关系记录，避免重复恢复或重放导致计数膨胀。

## 5. Mini-batch 学习流程

### 5.1 预测

批次开始时冻结 active 通用经验及其向量快照。所有样本共享该快照；同批新规则不可见。

### 5.2 保存证据

整批预测成功后按输入顺序揭示标签，并为每个样本写入 `CaseEvidence`。

### 5.3 正确样本

正确样本不额外调用 Qwen：

- 若注入经验的 sentiment 与真实标签一致，记录 support；
- 若不一致，记录 contradiction；
- 没有注入经验时只保存案例证据；
- 正确案例本身不自动生成候选规则。

### 5.4 错误样本

先执行确定性归因提示：

- 未检索经验：`missing_knowledge`；
- 检索经验均与真实标签冲突：`wrong_experience`；
- 存在正确标签经验但未进入 Top-K：后续版本处理，当前归为 `retrieval_failure` 的显式输入条件；
- 跨语言经验支持错误标签：`negative_transfer`；
- 其他：`reasoning_error`。

随后调用 Qwen 一次，输入案例、预测、真实标签、确定性提示和已检索经验摘要，输出严格 Attribution JSON。解析失败使用格式纠正提示重试；最终失败保存失败记录并使用保守的确定性归因，不中断整批学习。

### 5.5 形成和匹配候选规则

将 `candidate_rule + corrected_reason` 向量化，并在相同 sentiment 的 candidate/active 规则中检索。

满足以下条件时合并：

- cosine similarity 大于等于 `merge_similarity`；
- sentiment 一致；
- source 范围不冲突；
- language 范围允许扩展；
- error type 兼容。

否则创建新的 candidate。

本版本的“聚类”采用确定性的在线阈值聚合，不引入需要预先指定簇数的 K-Means。每个候选规则相当于一个持续更新的语义簇。

## 6. 生命周期

默认配置：

```yaml
generalization:
  merge_similarity: 0.85
  minimum_support: 2
  minimum_batches: 2
  maximum_contradiction_ratio: 0.20
  minimum_active_reliability: 0.60
```

状态规则：

- 新规则为 `candidate`；
- 支持数达到阈值、来自至少两个 batch、冲突率不超阈值时晋升 `active`；
- active 规则冲突率超过阈值时变为 `conflicted`；
- 冲突率超过 0.5 且至少有三条证据时变为 `suppressed`；
- 状态变化写入事件表，不删除历史。

可靠性：

```text
(support_count + 1) /
(support_count + contradiction_count + 2)
```

## 7. 推理检索

默认只检索 active 通用经验：

```text
score = semantic_similarity
      + language_match_weight
      + source_match_weight
      + reliability_weight
```

Prompt 仅注入：

```text
semantic
sentiment
rule
corrected_reason
scope
reliability
```

不注入原始案例、案例 ID、完整事件或原始归因响应。

当尚无 active 规则时，预测退化为固定 zero-shot Prompt。候选规则不会影响推理，因此规则必须经过跨批次验证后才能使用。

## 8. 配置与消融

新增：

```yaml
attribution:
  enabled: true
  llm_for_errors_only: true
  max_retries: 2

generalization:
  enabled: true
  merge_similarity: 0.85
  minimum_support: 2
  minimum_batches: 2
  maximum_contradiction_ratio: 0.20
  minimum_active_reliability: 0.60

retrieval:
  k: 3
```

实验模式：

- `case_memory`：旧案例经验；
- `attribution_only`：保存归因但不形成通用规则；
- `generalized_experience`：完整机制；
- `generalized_without_lifecycle`：规则生成后直接 active，用于消融。

V1 实现优先保证 `generalized_experience` 和关闭归因/通用化的基础消融，不一次实现全部配置模板。

## 9. 失败处理

- 归因 API 或解析最终失败不终止主实验；
- 保存 sample ID、异常类型、原始响应和重试次数到 `attribution_failures.jsonl`；
- 使用确定性 error type 和保守候选规则继续；
- SQLite 证据、归因、规则和事件在单样本事务中写入；
- 通用经验索引与数据库不一致时拒绝检索并给出明确错误；
- test/dev 不生成案例证据、归因或规则。

## 10. 实验产物

除现有文件外新增：

```text
attributions.jsonl
attribution_failures.jsonl
generalized_experiences.jsonl
experience_evolution_metrics.json
```

进化指标至少包括：

```text
case_count
error_case_count
attribution_count
candidate_count
active_count
conflicted_count
suppressed_count
support_count
contradiction_count
compression_ratio
attribution_api_calls
```

## 11. 实现边界

新增模块：

```text
src/sentiment_agent/
├─ evidence/
│  ├─ models.py
│  └─ repository.py
├─ attribution/
│  ├─ models.py
│  ├─ deterministic.py
│  └─ llm_attributor.py
└─ generalization/
   ├─ models.py
   ├─ repository.py
   ├─ matcher.py
   ├─ lifecycle.py
   └─ service.py
```

`SentimentAgent` 仍负责批次预测上下文；新的 `ExperienceEvolutionService` 负责反馈后的证据、归因、匹配、合并和生命周期，避免继续扩张 Agent 类。

## 12. 测试与验收

开发严格 TDD。默认测试使用 Fake LLM 和 Fake Embedding，不访问网络或真实密钥。

必须验证：

- 每个训练样本产生且只产生一条案例证据；
- dev/test 不写证据；
- 正确样本不调用归因 LLM；
- 错误样本生成 Attribution；
- 相似、同标签规则合并；
- 相似但标签冲突的规则不合并；
- 单批支持不能晋升；
- 跨两个 batch 达到支持阈值后晋升 active；
- 冲突证据降低可靠性并改变状态；
- 同批候选规则不进入同批 Prompt；
- 下一批只能检索 active 通用经验；
- 归因解析失败可回退且不终止实验；
- 完整离线预测—反馈—归因—合并—晋升—检索闭环通过；
- 现有 zero-shot、进度、结果保存和测试集隔离测试继续通过。

完成定义：全套离线测试通过，覆盖率不低于 85%，mini 数据上的显式在线 smoke run 可由用户选择执行，默认验证不产生 API 费用。
