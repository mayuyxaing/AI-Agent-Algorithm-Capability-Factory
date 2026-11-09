# 工作流 v2 离线验收

本目录均为 mock 后端的真实本地执行产物；用于验证工程流程，不是 GLM 效果实验。Mock 内部仍使用模板，虽然新的真实模型提示中不再提供完整模板。

- [完整报告](runs/20261109T123928Z-06495bf7/report.md)：包含训练画像、两跳检索路径、结构化预处理和超参数计划。
- [逐步记录](runs/20261109T123928Z-06495bf7/events.json)：生成上下文包含 interface_contract，不包含 interface_reference 或完整分类器模板。
- 修复演示：宿主在模型原始代码上显式注入未知类别缺陷，逻辑回归第一次 unknown_category 检查失败，第二次通过；原始代码、实际执行代码和来源摘要均保留。这次修复响应来自 mock，不冒充真实 GLM。
- [三组对照流程](benchmark/summary.md)：相同 baseline 数据、seed=42、三组隔离，生成次数上限一致。本次各组首次通过，指标相同，不能据此推断真实知识增强效果。
- v2 真实 GLM 结果待运行；之前的真实模型案例是 [v1](../glm/PROGRESS.md)，两代证据明确区分。

每个例子的 report.json 及 knowledge_graph.json 是机器可读证据，模型文件可用于本地预测。
