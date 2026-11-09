# 算法能力验证报告

- Run: `20261109T110253Z-8d8f8a20`
- 状态：passed
- 模式：mock / deterministic-templates-v1
- 故障注入：True
- 数据 SHA256：`876184b02cc537285eabc94ddfccda984f0ed35ba043b52e97b502a56b6be483`

## 需求

根据客户使用时长、月费用、客服次数、合同类型和支付方式预测客户流失。处理缺失值与类别不平衡，比较逻辑回归和随机森林，使用 ROC-AUC 和 F1 评价，提供统一预测接口，能够处理未知类别。

## 检索依据

- 客户流失预测能力与输入输出 — `churn_business.md`:5（Knowledge:11b9faf0a4a66e45a3a4）
- 评价指标与业务约束 — `churn_business.md`:14（Knowledge:8c8cd3a7810e4b6a5290）
- 缺失值和类别编码策略 — `churn_business.md`:8（Knowledge:78a1b7dd32ededc8b25a）
- churn_category_encoder — `historical_code.py`:5（Function:392229c89714ea7dd2cd）
- 候选算法及适用条件 — `churn_business.md`:11（Knowledge:4018437553590d695b11）
- 历史失败案例：未知类别 — `churn_business.md`:20（Knowledge:43937262a6fbd4c91315）

历史经验检索数：0

## 候选比较（验证集）

| 算法 | 通过 | 尝试次数 | ROC-AUC | F1 |
|---|---|---|---|---|
| forest | True | 1 | 0.7424865430622009 | 0.6387434554973822 |
| logistic | True | 2 | 0.7672323564593301 | 0.6358974358974359 |

## 方案依据与修复轨迹

### forest

与线性模型比较非线性关系和特征交互；限制树深度与并行度。

- attempt 0: passed=True, injected=False
  - code: `forest/attempt_0/algorithm.py`; SHA256: `359118f0cf8182b2bcbf1c4a55ae38dd9abc897cd6c19da0ea26ba4c763d2d8c`
  - checks: `{'probability_contract': True, 'missing_column_rejected': True, 'column_order': True, 'unknown_category': True, 'all_missing_values': True, 'repeat_prediction': True, 'model_reload': True, 'minimum_roc_auc': True, 'minimum_f1': True}`
  - explanation: Offline template generation
### logistic

可解释线性基线；标准化数值特征，使用类别权重处理不平衡。

- attempt 0: passed=False, injected=True
  - code: `logistic/attempt_0/algorithm.py`; SHA256: `152c1a38ea779098042dd3b4ce248d1b2173c50c754da6b32a23acfaa0bb79a2`
  - checks: `{'probability_contract': True, 'missing_column_rejected': True, 'column_order': True, 'unknown_category': False, 'all_missing_values': True, 'repeat_prediction': True, 'model_reload': True, 'minimum_roc_auc': True, 'minimum_f1': True}`
  - explanation: Offline template generation

```text
{"unknown_category": "ValueError: Found unknown categories ['__previously_unseen_contract__'] in column 0 during transform"} Failed checks: ['unknown_category']
```

- attempt 1: passed=True, injected=False
  - code: `logistic/attempt_1/algorithm.py`; SHA256: `19e7486786d2ffc1faa13a53e0b2bb17f652cb7e9565158cc3abb3d20d027e75`
  - checks: `{'probability_contract': True, 'missing_column_rejected': True, 'column_order': True, 'unknown_category': True, 'all_missing_values': True, 'repeat_prediction': True, 'model_reload': True, 'minimum_roc_auc': True, 'minimum_f1': True}`
  - explanation: Regenerated safe template with unknown-category handling; not an LLM repair.

## 最终测试集评估

选定算法：logistic

```json
{
  "checks": {
    "probability_contract": true,
    "missing_column_rejected": true,
    "column_order": true,
    "unknown_category": true,
    "all_missing_values": true,
    "repeat_prediction": true,
    "model_reload": true,
    "minimum_roc_auc": true,
    "minimum_f1": true
  },
  "check_errors": {},
  "executed": true,
  "worker_seconds": 2.2688,
  "python_peak_mb": 51.005,
  "wall_seconds": 3.1044,
  "metrics": {
    "roc_auc": 0.7828947368421052,
    "f1": 0.6432160804020101,
    "accuracy": 0.7041666666666667
  },
  "passed": true
}
```

## 解释边界

数据为固定种子的模拟数据，指标不能代表真实客户业务效果。按验证集选择，最终测试结果不参与修复。
Mock 的需求解析、规划、代码生成和修复是规则/模板行为，不能作为真实大模型效果证据。
角色按状态机顺序交接；未实现自主并行多智能体，也未实现操作系统级安全沙箱。
python_peak_mb 仅统计 Python 内存分配，非进程总内存；wall_seconds 包含进程启动。
