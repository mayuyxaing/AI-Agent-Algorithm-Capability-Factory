# 五分钟演示与答辩准备

1. 先说明：这是客户流失二分类能力工厂，不是通用 AutoML，也不是只调用一次 prompt 写代码。
2. 展示 materials/churn_business.md 和 historical_code.py：知识来自何处，字段、约束和处理经验是什么。
3. 执行 README 中三次 run：正常、注入故障并修复、历史经验复用。
4. 打开修复那次 report.md：初版 unknown_category 检查失败；下一轮通过；对比两个 algorithm.py 和 events.json。
5. 打开下一次 events.json：规划上下文和返回的 experience_ids 指向上次 FailureCase。说明检索复用已实现，但没有证明模型性能因此提高。
6. 展示验证与测试结果：验证集选择逻辑回归；测试集只评估最终代码，不用于改代码。
7. 用 scripts/predict.py 加载 selected 模型，对 examples/predict_input.csv 输出客户概率。
8. 打开 knowledge_graph.html 或 JSON，指出 Source、AlgorithmVersion、ValidationRun 和 FailureCase。

## 老师可能追问

**为什么不用 LangChain/Neo4j？** 当前规模下显式工作流和 SQLite 更方便复现与追溯；接口允许未来替换，不把框架数量当成研究贡献。

**mock 是否算用了大模型？** mock 不算。题目允许离线核心演示；代码另提供真实 Ollama 适配器。交付中的指标来自 mock 模板生成的算法，不能称为真实 LLM 实验结果。

**知识库到底有什么用？** 保存可引用的材料，进入方案及代码上下文；保留执行失败，供后续任务检索。需要对照实验才能判断是否提高首次通过率。

**下一步研究怎么做？** 固定任务集、同一模型和预算，比较直接生成、文档检索增强、文档加失败经验增强；记录首次通过率、最终通过率、修复次数、调用成本和性能。使用多个种子和独立测试集，禁止根据测试结果反复调方案。

**最大不足是什么？** 当前只支持固定字段的单任务；真实模型尚需接入环境实测；受限子进程不是安全沙箱；失败分析和离线修复仍使用规则。
