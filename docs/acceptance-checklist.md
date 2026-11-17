# 功能与要求对应表

我按照任务要求，将项目功能、实现位置和运行结果整理如下。

## 当前结论

我已完成基础功能和对应交付材料，8 项进阶要求均有对应实现；其中接口采用 CLI，验证任务限于已注册的两类二分类。客户流失有真实 GLM 闭环，文本有 mock 闭环和真实代码经助手接口补丁后的半自动闭环。真实 GLM 文本自主闭环及真实三组对照尚未成功完成。本次交付包含完整源码、分阶段 Git 历史和运行材料，已发布至 GitHub 仓库。

状态含义：**已实现**表示存在可核对代码及相关验证；**部分实现**表示只覆盖该要求的一部分；**未实现**表示没有对应能力；**待提交**表示本地交付存在，但远程提交未执行。

## 一、题目六项核心要求

| 编号 | 要求 | 状态与范围 | 实现或运行证据 |
|---|---|---|---|
| C01 | 使用开源模型或 LLM 接口 | 已实现；真实 GLM 接口有运行记录。Ollama/OpenAI 适配器不等于已在本机实测 | [LLM 适配器](../forge/llm.py)、[真实客户流失结果](../examples/v2/REAL-RESULTS.md) |
| C02 | 选择具体行业场景 | 已实现；客户流失预测、客服工单紧急程度分类；均使用模拟材料和模拟数据 | [客户材料](../materials/churn_business.md)、[文本材料](../materials/text_business.md)、[任务插件](../forge/tasks.py) |
| C03 | 能力知识库或知识图谱 | 已实现；SQLite 属性图，含输入输出、条件、指标、环境、版本与验证关系 | [schema](schema.md)、[存储与检索](../forge/knowledge.py)、[图谱样例](../examples/v3/text-assisted-repair/runs/20261115T045921Z-76a2bd42/knowledge_graph.json) |
| C04 | 自然语言到可运行算法的自动/半自动复刻 | 已实现；真实客户流失成功，真实文本为明确标注的辅助修复成功 | [客户流失报告](../examples/v2/real-success/report.md)、[文本辅助修复说明](../examples/v3/text-assisted-repair/README.md) |
| C05 | 统一验证正确性、指标、稳定性、接口 | 已实现；独立宿主计算指标，按任务选择检查，合格最终版本再评估测试集 | [验证入口](../forge/validation.py)、[执行器](../forge/worker.py)、[测试配置](../configs/text.json) |
| C06 | 简洁界面、CLI 或 API | 已实现 CLI；没有 Web 界面或 HTTP API。CLI 已满足题目中的任选要求 | [命令入口](../forge/cli.py)、[运行指南](../README.md) |

## 二、完整工作流 a–g

| 环节 | 状态 | 可核对行为 |
|---|---|---|
| a. 理解能力描述 | 已实现 | `understand` 返回任务、目标、约束、指标；宿主校验，不接受任意未知任务 |
| b. 检索知识 | 已实现 | 文档/函数及历史失败检索；Requirement → Strategy → Knowledge/Function 固定两跳路径 |
| c. 规划实现 | 已实现 | 候选算法、预处理、有界超参数、理由、证据 ID；结构及引用校验 |
| d. 生成代码 | 已实现 | 返回 `build_model` 和 `predict` 的完整源码；保留原始响应、源码和摘要 |
| e. 执行测试评估 | 已实现 | AST 检查、独立子进程、宿主指标、接口与模型重载检查 |
| f. 根据错误修复 | 已实现 | 代码生成/修复共用预算；默认首次生成加最多两轮修复，服务故障与代码缺陷分别处理 |
| g. 验证结果沉淀 | 已实现 | 保存运行、算法版本、环境、失败经验及关系；导出 JSON 和 HTML |

统一实现见 [workflow.py](../forge/workflow.py)；可沿着[真实客户流失 events.json](../examples/v2/real-success/events.json)核对角色输入输出。该真实案例包含宿主显式注入的未知类别缺陷，不是自然生成错误修复成功的证明。

## 三、基础交付清单

| 编号 | 交付要求 | 状态 | 对应文件 |
|---|---|---|---|
| B01 | 完整项目代码库 | 已具备本地 Git 仓库 | [包配置](../pyproject.toml)、[项目说明](../README.md) |
| B02 | 示例行业数据、文档或模拟材料 | 已提供 | [客户数据](../data/churn.csv)、[文本数据](../data/tickets.csv)、[历史示例代码](../materials/historical_code.py) |
| B03 | 知识图谱 schema 设计说明 | 已提供 | [schema.md](schema.md) |
| B04 | 知识抽取结果文件 | 已提供 JSON；图谱包含 Knowledge、Function、Source 等节点 | [抽取与回写样例](../examples/knowledge_graph.json)、[真实文本原始图谱](../examples/v3/text-assisted-repair/source-run/knowledge_graph.json) |
| B05 | Agent 工作流代码或配置 | 已提供 | [工作流](../forge/workflow.py)、[架构](architecture.md) |
| B06 | 至少一个完整生成示例 | 已提供，须说明运行模式 | [真实 GLM 客户流失](../examples/v2/real-success/report.md)、[文本半自动案例](../examples/v3/text-assisted-repair/README.md) |
| B07 | 自动验证脚本 | 已提供；流程内验证与独立重新验证均可用 | [validate.py](../scripts/validate.py)、[验证实现](../forge/validation.py) |
| B08 | 验证报告样例 | 已提供 Markdown/JSON | [客户报告](../examples/v2/real-success/report.json)、[文本报告](../examples/v3/text-assisted-repair/runs/20261115T045921Z-76a2bd42/report.md) |
| B09 | README.md | 已提供；覆盖题目要求的十项内容 | [README](../README.md) |
| B10 | 环境依赖文件 | 已提供范围依赖、开发依赖及历史锁定文件 | [requirements.txt](../requirements.txt)、[开发依赖](../requirements-dev.txt)、[环境锁定](../requirements-lock.txt) |
| B11 | 使用示例与测试用例 | 已提供；当前归档为 110 项 pytest 通过 | [测试结果](../examples/test-results.txt)、[JUnit](../examples/test-results.xml)、[测试代码](../tests/test_factory.py) |
| B12 | GitHub 仓库提交 | 已发布；源码和分阶段提交历史已上传 | [项目仓库](https://github.com/mayuyxaing/AI-Agent-Algorithm-Capability-Factory) |

源码压缩包、提交包及 Git history bundle 在项目旁的 outputs 目录提供；它们是打包产物，不要求进入源码 Git 仓库。打包检查包含 ZIP 完整性、报告对应的数据摘要和 Git bundle 可读性。

## 四、8 项进阶要求

| 编号 | 要求 | 状态与证据 | 实现范围 |
|---|---|---|---|
| A01 | Web、CLI 或 API 接口 | 已实现：[CLI](../forge/cli.py) | 提供 CLI，没有 Web/HTTP API |
| A02 | 多轮代码修复 | 已实现：[工作流](../forge/workflow.py)、[修复测试](../tests/test_v2.py)、[真实修复版本](../examples/v2/real-success/logistic/attempt_1/algorithm.py) | 支持配置修复预算；不保证所有模型每次都能修好 |
| A03 | 多候选自动比较 | 已实现：[客户与文本配置](../forge/tasks.py)、[文本完整报告](../examples/v3/text-assisted-repair/runs/20261115T045921Z-76a2bd42/report.json) | 只在通过验证的候选中，按配置指定的验证指标选择 |
| A04 | 图谱可视化 | 已实现：[HTML 生成](../forge/reporting.py)、[离线图谱](../examples/v3/text-assisted-repair/runs/20261115T045921Z-76a2bd42/knowledge_graph.html) | SVG 关系图、悬停信息和节点表；不是图数据库管理界面 |
| A05 | 自动生成验证报告 | 已实现：[报告模块](../forge/reporting.py)、[报告样例](../examples/v2/real-success/report.md) | 自动生成 Markdown、JSON 及逐步轨迹 |
| A06 | 验证结果回写知识库 | 已实现：[save_run](../forge/knowledge.py)、[回写图谱](../examples/v3/text-assisted-repair/runs/20261115T045921Z-76a2bd42/knowledge_graph.json) | 保存版本、运行、指标、环境及失败关系；支持后续检索 |
| A07 | 不同任务验证配置 | 已实现：[churn.json](../configs/churn.json)、[text.json](../configs/text.json)、[跨任务测试](../tests/test_task_plugins.py) | 支持两类已注册二分类任务，不能直接处理任意字段/任务 |
| A08 | 算法模板或指标插件扩展 | 已实现：[插件契约](plugin-tasks.md)、[算法元数据](../forge/plugins/text_nb.json)、[扩展测试](../tests/test_task_plugins.py) | 注册任务、算法元数据/安装入口及指标函数；新任务仍需实现验证契约 |

文本案例的 `selection_metric` 为 F1；两个候选的验证 F1 相同，现实现保留候选顺序中的第一个最大值，即 text_logistic。文本模型按 F1 选择，客户流失模型按 ROC-AUC 选择。

## 五、10 项加分要求

| 编号 | 要求 | 核对结果 | 证据及限制 |
|---|---|---|---|
| P01 | 图搜索、Beam Search 或 MCTS 方案搜索 | 部分实现 | [固定两跳图检索](../forge/knowledge.py)属于检索；没有 Beam/MCTS 或搜索式算法规划，不能宣称已完整实现这一加分项 |
| P02 | 多智能体协作 | 未实现自主多智能体 | [串行多角色状态机](architecture.md)共享 LLM 后端；多次调用和多个角色不等于自主协商/并行多智能体 |
| P03 | 从真实仓库抽取函数、依赖、能力 | 部分实现 | [AST 抽取](../forge/knowledge.py)可递归扫描本地目录，已验证的是模拟历史代码；没有外部真实仓库提取案例、类方法/调用图/完整依赖解析 |
| P04 | 安全检查和沙箱执行 | 部分实现 | [AST 检查](../forge/safety.py)、[子进程执行](../forge/validation.py)及超时/线程限制；没有 OS 级文件、网络、进程隔离，不能称安全沙箱 |
| P05 | 自动分析失败并形成经验 | 已实现规则版本 | [错误分类](../forge/failures.py)、[经验回写](../forge/knowledge.py)、[文本错误测试](../tests/test_task_plugins.py)；不是通用模型根因推断，未证明复用带来统计提升 |
| P06 | 算法能力版本管理 | 已实现基础版本 | 每轮源码摘要、attempt、run、指标、环境及 AlgorithmVersion 关系；没有发布版本审批、回滚或迁移服务 |
| P07 | 自然语言说明设计依据 | 已实现 | [真实计划与解释](../examples/v2/real-success/report.md)、[生成轨迹](../examples/v2/real-success/events.json)；引用存在和原文匹配不等于解释必然正确 |
| P08 | 跨场景迁移 | 已实现限定范围 | [两任务插件](../forge/tasks.py)、[文本完整半自动结果](../examples/v3/text-assisted-repair/README.md)；数据均模拟，文本自主模型生成尚未通过 |
| P09 | 自动接口文档或部署配置 | 部分实现 | [自动 CLI/插件契约目录](generated-interfaces.md)可生成；没有 HTTP OpenAPI、容器或线上部署配置 |
| P10 | 性能优化和资源消耗分析 | 部分实现 | 报告有 worker/wall 时间、Python 分配峰值和线程限制；没有优化前后对照，峰值不是 RSS，没有系统级 CPU/内存基准 |

## 六、开发要求核对

| 要求 | 当前状态 |
|---|---|
| Git 与清晰提交信息 | 本地 Git 分阶段提交；提交包附完整 history bundle |
| 项目说明、环境、运行、示例文档 | README 与 docs 已提供；最终演示稿在后续准备步骤更新 |
| 结构清晰、模块化、注释 | 数据、任务、知识、规划、模型、验证、报告、入口分模块；不能仅凭此保证所有代码质量评分 |
| 本地可复现 | 已完成离线测试与提交包解压后的辅助修复复现；真实 API 调用受服务可用性影响 |
| Python 主语言 | 已采用 Python |
| 技术选择理由 | [架构说明](architecture.md)：SQLite、显式工作流、sklearn、CLI 的选择与限制 |
| 无大模型时可 mock | 显式 mock 可演示；真实、mock、辅助修复模式分别保留，不混为同一证据 |

## 七、README 十项内容索引

| 文档内容 | 对应位置 |
|---|---|
| 1. 背景和目标 | README 第 1 节 |
| 2. 架构和模块 | README 第 2 节、architecture.md |
| 3. 图谱 schema 和示例 | README 第 6 节、schema.md、归档图谱 |
| 4. Agent 工作流 | README 第 2、7 节、workflow.py |
| 5. 配置和运行 | README 第 3–5 节 |
| 6. 数据与测试任务 | README 第 6 节、data 与 materials |
| 7. 生成代码示例 | README 第 7 节及生成的 algorithm.py |
| 8. 验证结果与报告 | README 第 8 节、examples 中的分类证据 |
| 9. 挑战和解决办法 | README 第 9 节 |
| 10. 可扩展方向 | README 第 10 节、plugin-tasks.md |

## 八、项目概述

> 我实现了一个基于 LLM、多角色工作流和 SQLite 属性图的算法能力工厂原型，覆盖能力抽取、知识检索、方案规划、代码生成、独立验证、有限修复和结果回写。客户流失任务有真实 GLM 运行记录，其中修复演示的缺陷由宿主显式注入；文本分类通过任务插件接入，真实模型代码经明确标注的助手接口补丁后完成半自动验证。系统还支持多候选比较、离线图谱视图、版本追溯及规则式失败经验复用。

当前实现采用串行多角色调度和子进程执行，暂未加入自主多智能体、MCTS 与操作系统级沙箱。真实行业数据和真实模型三组对照是后续实验方向。

## 九、后续工作

后续我将优先开展固定任务集上的真实模型对照实验，统计首次通过率、修复成本和最终通过率；再完善生成代码执行隔离，并扩展任务与算法插件。报告摘要、文件摘要及测试计数见 [acceptance-evidence.json](acceptance-evidence.json)。

## 验证与状态管理补充

我增加了逐类别字段的未知值检查、文档更新和删除后的生效快照，以及对照单元重启隔离与完整性核对。实现与边界见 [修正说明](review-fixes.md)，回归用例见 [test_review_regressions.py](../tests/test_review_regressions.py)。历史模型报告维持原来的检查集合，当前检查以新运行结果为准。

新协议的离线三组运行记录：[summary.md](../examples/review/benchmark-v3/summary.md)、[manifest.json](../examples/review/benchmark-v3/manifest.json)。
