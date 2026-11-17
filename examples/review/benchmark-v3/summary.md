# 对照实验结果

模式：mock / deterministic-templates-v1；状态：complete

Mock 只验证实验流程，不能据此证明真实 LLM 的知识增强效果。各组每候选生成次数上限相同；直接生成/知识增强组失败后独立重采样，不接收代码或验证反馈；修复组接收反馈。首次通过即停止，实际调用和 tokens 仍可能不同。
语料固定为规则/AST 抽取，不把 LLM 抽取的差异混入本次对照。服务错误单独标注，未运行的单元不计为算法失败。

| 条件 | 组别 | 状态 | 首次通过候选/总候选 | 修复调用 | 已报告 tokens | 测试 AUC |
|---|---|---|---|---|---|---|
| baseline / 42 | direct | passed | 1/2 | 0 | 0 | 0.8489010989010989 |
| baseline / 42 | knowledge | passed | 1/2 | 0 | 0 | 0.8489010989010989 |
| baseline / 42 | knowledge_repair | passed | 1/2 | 2 | 0 | 0.8489010989010989 |

相同 case/seed 使用相同数据与划分；不同组数据库隔离，不共享失败经验。
每个单元仅对选中版本评估保留测试集；不根据组间测试结果修改方案。
续跑先核对代码、插件、依赖、数据及已完成报告。服务中断单元从新数据库完整重启，不继承响应缓存和失败经验；原尝试保留并计入成本。生成预算按每次尝试计算，重启不等于原单元内部断点续接。独立重复实验使用新输出目录。

## 按组汇总

```json
{
  "direct": {
    "evaluated_tasks": 1,
    "provider_interrupted_tasks": 0,
    "task_success_rate": 1.0,
    "candidate_first_pass_rate": 0.5,
    "repair_calls": 0,
    "reported_tokens_including_prior_attempts": 0
  },
  "knowledge": {
    "evaluated_tasks": 1,
    "provider_interrupted_tasks": 0,
    "task_success_rate": 1.0,
    "candidate_first_pass_rate": 0.5,
    "repair_calls": 0,
    "reported_tokens_including_prior_attempts": 0
  },
  "knowledge_repair": {
    "evaluated_tasks": 1,
    "provider_interrupted_tasks": 0,
    "task_success_rate": 1.0,
    "candidate_first_pass_rate": 0.5,
    "repair_calls": 2,
    "reported_tokens_including_prior_attempts": 0
  }
}
```

成功率分母排除服务中断，包含模型结构化契约失败和算法验证失败；单案例/单种子不能支持统计显著性结论。
