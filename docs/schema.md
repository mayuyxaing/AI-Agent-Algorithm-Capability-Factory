# 能力知识图谱 schema v1.1

采用 SQLite 属性图：nodes(id, kind, title, properties JSON)，edges(source, relation, target)，runs(id, report JSON)，ingestions(id, report JSON)。关系两端有外键；同一三元组去重。完整图可导出 JSON，离线 HTML 为辅助视图。没有依赖外部图数据库。

| 节点类型 | 关键属性 | 含义 |
|---|---|---|
| Capability | task, input, output, conditions, dependencies | 任务及接口契约 |
| Source | path, sha256, corpus | 不可变内容摘要与语料目录标识的来源版本 |
| Knowledge | text, evidence_quote, source_id, source_path, line, extraction | 文档知识及可核对引文 |
| Function | text, dependencies, evidence_quote, source_path, line | Python AST 提取的函数和导入依赖 |
| AlgorithmVersion | sha256, attempt, path, run_id, passed, validation_metrics, checks | 每次生成/修复的代码版本 |
| Metric | task, title | 指标定义标识；实现位于独立指标注册表 |
| Environment | python, platform, numpy, pandas, scikit-learn, joblib | 实际运行依赖版本 |
| ValidationRun | status, mode, model, test_metrics, dataset_sha256, seed, report_ref | 测试指标及完整报告引用 |
| FailureCase | task, error, checks, injected, resolved_in_run, remedy | 失败、修复建议及演示故障标记 |
| Requirement | task, key, extraction=rule_index | 缺失值、未知类别、不平衡等条件 |
| Strategy | task, key, extraction=rule_index | 对应处理策略，链接支持原文 |

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
| ADDRESSED_BY | Requirement → Strategy |
| SUPPORTED_BY | Strategy → Knowledge/Function |

## 身份与版本策略

来源和知识节点默认使用标题及属性内容 SHA256 的前 20 位生成 ID，相同标题与属性重复抽取幂等；来源改变会保留新版本。能力 ID 为 capability:<task_id>，如 capability:churn_classification 和 capability:text_classification。算法版本以 run、候选 ID、attempt 区分，并保存完整代码 SHA256。完整验证记录保存在 runs 表及每次运行 report.json 中。

材料相对路径保留，语料身份使用规范绝对目录，不同目录独立管理。移动材料目录会被视为新语料。代码路径相对于该次运行根目录。数据内容摘要、随机种子与环境版本共同支持追溯；不保证跨平台浮点结果逐位一致。

`source_versions(source_id, corpus, path)` 记录来源归属；`active_facts(corpus, fact_id)` 记录当前生效快照。一次完整抽取成功后，在一个事务内替换该目录的生效事实集合。文档更新、删除或抽取中断后，旧节点仍可追溯，默认词法和图谱检索只读取当前快照。schema 1.1 导出包含 `retrieval_state`，使用 `import_graph()` 恢复可保留生效状态。旧格式迁移方式见 [修正说明](review-fixes.md)。

## 抽取与检索边界

离线模式按 Markdown 二级标题分段，并从 Python AST 抽取顶层函数、docstring、函数源码与模块导入。它不是语义抽取模型。真实模型抽取额外增加 LLM facts，推荐通过 source_chunk_id 引用宿主预先编号的原文片段，由宿主附加准确原文及行号；也兼容旧版精确 evidence_quote。未知片段 ID、冲突引文和不合规事实拒绝入库，保存在 ingestions 审计表及可选审计 JSON 中。默认继续使用已核对知识，并显式报告 partial_llm 和拒绝数量；--strict-extraction 可在任一拒绝时停止。模型格式不合规可纠正一次，仍失败则停止；网络及服务错误独立处理。原文引证核对不等于知识真实性或语义支持关系已被证明。

v2 根据训练画像和显式约束，从 Requirement 经 Strategy 遍历至支持的 Knowledge/Function，结合英文词和中文双字词重叠排序，返回前 6 条及 retrieval_paths。每条路径可在 edges 表核验。关系采用显式关键词规则索引，不等于模型已证明语义支持关系，也不是通用图搜索。历史 FailureCase 按任务过滤，作为独立上下文交给规划与生成角色。

适用条件的完整描述仍保存在 Knowledge 中；v2 单独索引了 Requirement 与 Strategy 节点。后续可增加字段级 DataRequirement、关系约束、跨目录语料迁移与人工审核。
