# 真实 GLM 实测进展

以下来自本机真实 glm-4.7-flash 调用，和 mock 示例分开保存。

- [连接测试](connection-check.json)：需求理解请求通过，返回任务 churn_classification；记录有服务端响应 ID、usage 和耗时。
- [知识抽取审计](ingestion-report.json)：LLM 接受 12 条、拒绝 0 条；加上规则/AST 抽取共 19 条。
- [知识图谱快照](knowledge_graph.json)：保留来源引用及抽取方式。
- [完整流程尝试](failed-attempt/report.md)：状态 failed，在需求理解步骤遇到服务端拥堵 1305；尚未生成候选代码，也没有真实 GLM 算法指标。

失败记录保留原样，不改成通过。已增加知识复用、模型响应检查点和有限拥堵重试；只有后续实际生成并验证通过的报告才能作为完整闭环证据。
