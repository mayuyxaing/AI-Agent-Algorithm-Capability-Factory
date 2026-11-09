# 实际运行结果

三次运行均使用 mock 规则/模板后端和同一模拟数据集。真实 LLM 未实测；重复运行不构成独立统计实验。

| 示例 | 状态 | 选中算法 | 测试 ROC-AUC | 测试 F1 | 检索历史失败数 |
|---|---|---|---|---|---|
| [正常生成](runs/20261109T110204Z-18c6ac07/report.md) | passed | logistic | 0.7829 | 0.6432 | 0 |
| [故障注入与修复](runs/20261109T110253Z-8d8f8a20/report.md) | passed | logistic | 0.7829 | 0.6432 | 0 |
| [历史经验复用](runs/20261109T110613Z-9556abfb/report.md) | passed | logistic | 0.7829 | 0.6432 | 1 |

## 候选比较（正常运行的验证集）

| 算法 | ROC-AUC | F1 |
|---|---|---|
| forest | 0.7425 | 0.6387 |
| logistic | 0.7672 | 0.6359 |

## 修复与复用证据

- 第二次运行显式注入未知类别故障；逻辑回归 attempt_0 的 unknown_category 检查失败，attempt_1 通过。Mock 修复采用安全模板再生成。
- 第三次运行检索上次 FailureCase，规划产物 experience_ids 引用该记录，生成上下文也包含这条经验。
- 正常常数概率基线的验证 ROC-AUC 为 0.5。测试指标未参与候选选择或后续修复。
- 单次运行耗时与 Python 分配峰值见 report.json，非严谨性能基准。

完整最新图谱：[JSON](knowledge_graph.json) / [离线 HTML](knowledge_graph.html)。
预测结果示例：[predictions.csv](predictions.csv)。测试执行记录：[test-results.txt](test-results.txt)。
