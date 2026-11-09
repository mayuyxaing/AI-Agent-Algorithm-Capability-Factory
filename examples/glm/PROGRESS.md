# 真实 GLM 实测结果

模型为 glm-4.7-flash，运行 20261109T121459Z-9d94bcde。所有模型响应均来自本次实际 API 调用，checkpoint_hits 为空；本轮复用了先前成功的知识抽取。

## 已完成的证据

- [连接测试](connection-check.json)：真实需求理解请求通过。
- [知识抽取](ingestion-report.json)：LLM 接受 12 条、拒绝 0 条；加上规则/AST 共 19 条。
- [完整验证报告](successful-run/report.md) / [机器可读报告](successful-run/report.json)：status=passed。
- [角色输入输出](successful-run/events.json)：understand、plan、generate×2 的真实交接。
- [最终生成代码](successful-run/selected/algorithm.py) / [预测示例](predictions.csv)：保存的模型已加载并对三位新客户预测，含未知类别和缺失值。
- [更新的知识图谱](knowledge_graph.json)：本次候选版本、验证结果与来源关系已回写。
- [之前的失败记录](failed-attempt/report.md)：1305 拥堵失败原样保留，不冒充通过。

## 候选比较

| 候选 | 验证 ROC-AUC | 验证 F1 | 尝试次数 | 通过 |
|---|---|---|---|---|
| forest | 0.7425 | 0.6387 | 1 | 是 |
| logistic | 0.7672 | 0.6359 | 1 | 是 |

按验证集 ROC-AUC 选择 logistic，随后只对选中版本评估测试集。测试 ROC-AUC=0.7829，F1=0.6432，accuracy=0.7042。最终 9 项检查全部通过。常数概率基线的验证 ROC-AUC=0.5。

本次成功的 4 个推理响应报告用量合计 7881 tokens；不包含先前连接、知识抽取及没有返回 usage 的失败请求，不能直接用来宣称总账单或免费政策。

## 解释边界

- 数据为模拟数据；这是一条工程示例，不是独立任务集上的效果实验。
- 生成上下文提供了完整接口模板，实际结果与模板高度一致。因此展示的是有模板参考的真实 LLM 算法复刻，不是从零发明算法。与 mock 指标相同并不能证明模型更强。
- 两个候选首次通过，本轮没有调用 repair。修复闭环的演示证据仍是此前明确标注的 mock 故障注入，不声称真实 GLM 已演示修复。
- 本轮检索前 6 条均来自规则/AST 产物；LLM 抽取事实已入库，但未进入本轮检索结果。没有证明 LLM 抽取提高生成效果。
- task_spec.constraints 返回了占位值 ["strings"]，没有完整提取自然语言约束。实际算法通过了独立配置的缺失值/未知类别等检查；这不等于需求解析字段已完全正确，后续应增加结构化语义验收。
