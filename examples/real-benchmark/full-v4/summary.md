# 对照实验结果

模式：glm / glm-4.7-flash；状态：complete

Mock 只验证实验流程，不能据此证明真实 LLM 的知识增强效果。各组每候选生成次数上限相同；直接生成/知识增强组失败后独立重采样，不接收代码或验证反馈；修复组接收反馈。首次通过即停止，实际调用和 tokens 仍可能不同。
语料固定为规则/AST 抽取，不把 LLM 抽取的差异混入本次对照。服务错误单独标注，未运行的单元不计为算法失败。

| 条件 | 组别 | 状态 | 首次通过候选/总候选 | 修复调用 | 已报告 tokens | 测试 AUC |
|---|---|---|---|---|---|---|
| baseline / 42 | direct | passed | 0/2 | 0 | 13423 | 0.8574448807006947 |
| baseline / 42 | knowledge | passed | 1/2 | 0 | 20870 | 0.8574448807006947 |
| baseline / 42 | knowledge_repair | passed | 2/2 | 0 | 11762 | 0.879492600422833 |
| baseline / 43 | direct | passed | 0/2 | 0 | 11985 | 0.7912679425837321 |
| baseline / 43 | knowledge | passed | 2/2 | 0 | 11674 | 0.7912679425837321 |
| baseline / 43 | knowledge_repair | failed | 0/0 | 0 | 8052 | N/A |
| baseline / 44 | direct | failed | 0/0 | 0 | 4946 | N/A |
| baseline / 44 | knowledge | failed | 0/0 | 0 | 7927 | N/A |
| baseline / 44 | knowledge_repair | passed | 1/2 | 1 | 17047 | 0.8356991845363939 |
| missing_high / 42 | direct | failed | 0/0 | 0 | 4655 | N/A |
| missing_high / 42 | knowledge | passed | 1/2 | 0 | 18882 | 0.7958320749018424 |
| missing_high / 42 | knowledge_repair | failed | 0/0 | 0 | 7995 | N/A |
| missing_high / 43 | direct | passed | 1/2 | 0 | 9958 | 0.7819976076555023 |
| missing_high / 43 | knowledge | passed | 1/2 | 0 | 15146 | 0.7819976076555023 |
| missing_high / 43 | knowledge_repair | failed | 0/0 | 0 | 7962 | N/A |
| missing_high / 44 | direct | failed | 0/0 | 0 | 4507 | N/A |
| missing_high / 44 | knowledge | passed | 1/2 | 0 | 15187 | 0.7774086378737541 |
| missing_high / 44 | knowledge_repair | passed | 0/2 | 3 | 25571 | 0.7671398369072788 |
| minority / 42 | direct | passed | 1/2 | 0 | 8255 | 0.7956989247311828 |
| minority / 42 | knowledge | passed | 1/2 | 0 | 19295 | 0.802469135802469 |
| minority / 42 | knowledge_repair | failed | 0/0 | 0 | 7923 | N/A |
| minority / 43 | direct | no_valid_candidate | 0/2 | 0 | 14558 | N/A |
| minority / 43 | knowledge | passed | 1/2 | 0 | 23244 | 0.8336842105263158 |
| minority / 43 | knowledge_repair | failed | 0/0 | 0 | 7996 | N/A |
| minority / 44 | direct | failed | 0/0 | 0 | 4407 | N/A |
| minority / 44 | knowledge | passed | 0/2 | 0 | 23611 | 0.7925347222222223 |
| minority / 44 | knowledge_repair | passed | 0/2 | 4 | 36617 | 0.7925347222222223 |

相同 case/seed 使用相同数据与划分；不同组数据库隔离，不共享失败经验。
每个单元仅对选中版本评估保留测试集；不根据组间测试结果修改方案。
续跑先核对代码、插件、依赖、数据及已完成报告。服务中断单元从新数据库完整重启，不继承响应缓存和失败经验；原尝试保留并计入成本。生成预算按每次尝试计算，重启不等于原单元内部断点续接。独立重复实验使用新输出目录。

## 按组汇总

```json
{
  "direct": {
    "evaluated_tasks": 9,
    "provider_interrupted_tasks": 0,
    "task_success_rate": 0.4444444444444444,
    "candidate_first_pass_rate": 0.2,
    "repair_calls": 0,
    "reported_tokens_including_prior_attempts": 77147
  },
  "knowledge": {
    "evaluated_tasks": 9,
    "provider_interrupted_tasks": 0,
    "task_success_rate": 0.8888888888888888,
    "candidate_first_pass_rate": 0.5,
    "repair_calls": 0,
    "reported_tokens_including_prior_attempts": 195792
  },
  "knowledge_repair": {
    "evaluated_tasks": 9,
    "provider_interrupted_tasks": 0,
    "task_success_rate": 0.4444444444444444,
    "candidate_first_pass_rate": 0.375,
    "repair_calls": 8,
    "reported_tokens_including_prior_attempts": 147031
  }
}
```

成功率分母排除服务中断，包含模型结构化契约失败和算法验证失败；单案例/单种子不能支持统计显著性结论。
