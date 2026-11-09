# 能力知识图谱 schema v1

采用 SQLite 属性图：nodes(id, kind, title, properties JSON)，edges(source, relation, target)，runs(id, report JSON)。关系两端有外键；同一三元组去重。完整图可导出 JSON，离线 HTML 为辅助视图。没有依赖外部图数据库。

| 节点类型 | 关键属性 | 含义 |
|---|---|---|
| Capability | task, input, output, conditions, dependencies | 任务及接口契约 |
| Source | path, sha256 | 不可变内容摘要标识的来源版本 |
| Knowledge | text, evidence_quote, source_id, source_path, line, extraction | 文档知识及可核对引文 |
| Function | text, dependencies, evidence_quote, source_path, line | Python AST 提取的函数和导入依赖 |
| AlgorithmVersion | sha256, attempt, path, run_id, passed, validation_metrics, checks | 每次生成/修复的代码版本 |
| Metric | task, title | 指标定义标识；实现位于独立指标注册表 |
| Environment | python, platform, numpy, pandas, scikit-learn, joblib | 实际运行依赖版本 |
| ValidationRun | status, mode, model, test_metrics, dataset_sha256, seed, report_ref | 测试指标及完整报告引用 |
| FailureCase | task, error, checks, injected, resolved_in_run, remedy | 失败、修复建议及演示故障标记 |

| 关系 | 起点 → 终点 |
|---|---|
| SUPPORTED_BY | Knowledge/Function → Source |
| IMPLEMENTED_BY | Capability → AlgorithmVersion |
| EVALUATED_BY | Capability → Metric |
| VALIDATED_IN | AlgorithmVersion → ValidationRun |
| EXECUTED_IN | ValidationRun → Environment |
| USED_KNOWLEDGE | ValidationRun → Knowledge/Function |
| EXPOSED | AlgorithmVersion → FailureCase |
| REUSED_EXPERIENCE | ValidationRun → FailureCase |

## 身份与版本策略

来源和知识节点默认使用内容 SHA256 的前 20 位生成 ID，相同材料重复抽取幂等；来源改变会保留新版本。能力 ID 固定为 capability:churn。算法版本以 run、候选 ID、attempt 区分，并保存完整代码 SHA256。完整验证记录保存在 runs 表及每次运行 report.json 中。

目录移动不影响材料相对路径。代码路径相对于该次运行根目录。数据内容摘要、随机种子与环境版本共同支持追溯；不保证跨平台浮点结果逐位一致。

## 抽取与检索边界

离线模式按 Markdown 二级标题分段，并从 Python AST 抽取顶层函数、docstring、函数源码与模块导入。它不是语义抽取模型。Ollama 抽取另增加 LLM facts，每条必须有原文连续子串 evidence_quote；不匹配则拒绝入库。原文引证核对不等于知识真实性已验证。

检索使用英文词及中文双字词重叠，返回带来源的前 6 条。这里是可检查的小型知识库检索，不是向量检索，也没有实现图路径搜索。历史 FailureCase 按任务过滤，作为独立结构化上下文交给规划与生成角色。

本版将算法适用条件、策略和环境要求保存在 Knowledge 属性中；只将能力、指标、环境、代码版本和验证结果单独建节点。后续可把 Strategy 和 DataRequirement 拆成独立实体，增加关系约束与人工知识审核。
