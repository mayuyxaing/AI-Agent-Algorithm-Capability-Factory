# 算法能力验证报告

- Run: `20261109T121459Z-9d94bcde`
- 状态：passed
- 模式：glm / glm-4.7-flash
- 故障注入：False
- 数据 SHA256：`876184b02cc537285eabc94ddfccda984f0ed35ba043b52e97b502a56b6be483`

## 需求

根据客户使用时长、月费用、客服次数、合同类型和支付方式预测客户流失。处理缺失值与类别不平衡，比较逻辑回归和随机森林，使用 ROC-AUC 和 F1 评价，提供统一预测接口，能够处理未知类别。

## 检索依据

知识抽取审计：status=passed，LLM 接受 12 条，拒绝 0 条；不合格事实未进入能力图谱。

复用了 0 次历史模型响应，来源运行见 report.json；算法验证结果没有缓存。

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
| logistic | True | 1 | 0.7672323564593301 | 0.6358974358974359 |

## 方案依据与修复轨迹

### forest

历史失败案例表明，类别编码器在遇到未知类别时会导致错误。随机森林适合捕捉非线性关系和特征交互，同时通过限制树深度和并行度控制过拟合和资源消耗。

- attempt 0: passed=True, injected=False
  - code: `forest/attempt_0/algorithm.py`; SHA256: `2e6abaf0d56ef067ccfae45fbd71a3c8ea37ab77a3f1d6720babfc23b491126a`
  - checks: `{'probability_contract': True, 'missing_column_rejected': True, 'column_order': True, 'unknown_category': True, 'all_missing_values': True, 'repeat_prediction': True, 'model_reload': True, 'minimum_roc_auc': True, 'minimum_f1': True}`
  - explanation: The code implements the `build_model` and `predict` functions using a scikit-learn Pipeline. `build_model` creates a preprocessing pipeline with median imputation and scaling for numeric features, and most-frequent imputation with One-Hot encoding (with `handle_unknown='ignore'`) for categorical features, followed by a RandomForestClassifier configured to handle class imbalance. `predict` validates the input frame contains all required features and returns the probability of the positive class (churn=1).
### logistic

逻辑回归提供可解释的线性基线，适合作为比较基准。通过标准化数值特征和使用类别权重处理不平衡，可以平衡模型性能和解释性。

- attempt 0: passed=True, injected=False
  - code: `logistic/attempt_0/algorithm.py`; SHA256: `7ed6327e83e98c04f829bc9910ce7e0be3beba8a88efa10518a771b2b9bdc5ee`
  - checks: `{'probability_contract': True, 'missing_column_rejected': True, 'column_order': True, 'unknown_category': True, 'all_missing_values': True, 'repeat_prediction': True, 'model_reload': True, 'minimum_roc_auc': True, 'minimum_f1': True}`
  - explanation: The model implements a Logistic Regression classifier using a scikit-learn Pipeline to handle preprocessing and classification. Numeric features (tenure, monthly_charges, support_calls) are imputed with the median and scaled using StandardScaler. Categorical features (contract, payment_method) are imputed with the most frequent value and encoded using OneHotEncoder with `handle_unknown='ignore'` to prevent errors on unseen categories. The Logistic Regression uses `class_weight='balanced'` to address potential class imbalance and `max_iter=1000` to ensure convergence. The `predict` function validates that all required columns are present in the input frame and returns the probability of the positive class (churn=1).

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
  "worker_seconds": 1.8802,
  "python_peak_mb": 51.006,
  "wall_seconds": 2.3284,
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
