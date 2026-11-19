# 算法能力验证报告

- Run: `20261117T164619Z-2b74b074`
- 状态：passed
- 模式：glm / glm-4.7-flash
- 故障注入：False
- 数据 SHA256：`23c681619201b59f721b4b7f1b6662c908bd945db5798381c0ef968df86a1681`

## 需求

客户流失预测，处理缺失值、未知类别和类别不平衡，比较逻辑回归和随机森林，使用 ROC-AUC 和 F1 评价。

## 检索依据

知识抽取审计：status=passed，LLM 接受 0 条，拒绝 0 条；不合格事实未进入能力图谱。

工作流 v3；实验组=knowledge_repair；每候选生成预算=3；未向真实模型提供完整算法模板。

## 训练数据画像与实现计划

```json
{
  "data_profile": {
    "scope": "training_only",
    "rows": 360,
    "columns": {
      "tenure": {
        "kind": "numeric",
        "missing_rate": 0.033333,
        "cardinality": 71,
        "constant": false,
        "quantiles": {
          "0.25": 20.0,
          "0.5": 36.5,
          "0.75": 55.25
        }
      },
      "monthly_charges": {
        "kind": "numeric",
        "missing_rate": 0.016667,
        "cardinality": 354,
        "constant": false,
        "quantiles": {
          "0.25": 41.53662006741646,
          "0.5": 68.45100204378839,
          "0.75": 95.30866672610692
        }
      },
      "support_calls": {
        "kind": "numeric",
        "missing_rate": 0.027778,
        "cardinality": 7,
        "constant": false,
        "quantiles": {
          "0.25": 1.0,
          "0.5": 1.0,
          "0.75": 2.0
        }
      },
      "contract": {
        "kind": "categorical",
        "missing_rate": 0.016667,
        "cardinality": 3,
        "constant": false
      },
      "payment_method": {
        "kind": "categorical",
        "missing_rate": 0.025,
        "cardinality": 2,
        "constant": false
      }
    },
    "positive_rate": 0.3638888888888889,
    "minority_rate": 0.3638888888888889,
    "requirements": [
      "train_only_preprocessing",
      "unknown_categories",
      "missing_values",
      "class_imbalance"
    ]
  },
  "plan": {
    "candidates": [
      {
        "algorithm": "logistic",
        "rationale": "逻辑回归作为线性基线模型，适合处理具有缺失值和类别不平衡的表格数据。结合标准化处理和 balanced 权重，可以有效捕捉主要特征与流失率之间的线性关系，提供良好的可解释性。",
        "hyperparameters": {
          "C": 1.0,
          "max_iter": 1000,
          "class_weight": "balanced",
          "solver": "lbfgs"
        }
      },
      {
        "algorithm": "forest",
        "rationale": "随机森林能够捕捉特征间的非线性交互作用，适合在客户流失数据中寻找复杂的决策模式。通过限制树深度和叶子节点样本数，可以有效控制过拟合，并在类别不平衡情况下通过 balanced 权重提升模型性能。",
        "hyperparameters": {
          "n_estimators": 100,
          "max_depth": 7,
          "min_samples_leaf": 5,
          "class_weight": "balanced"
        }
      }
    ],
    "preprocessing": {
      "numeric_imputer": "median",
      "categorical_imputer": "most_frequent",
      "scale_numeric": true,
      "unknown_categories": "ignore"
    },
    "evidence_ids": [
      "Knowledge:b1bde62ca7719f21fe42",
      "Knowledge:ee594e7236a1cbe8c520",
      "Knowledge:b8867b632089cbc5844c",
      "Function:abe0662caba9d48ee2eb"
    ],
    "experience_ids": [],
    "experience_applied": false
  }
}
```

- 缺失值和类别编码策略 — `churn_business.md`:8（Knowledge:b1bde62ca7719f21fe42）
  - 图谱路径：requirement:missing_values → strategy:imputation → Knowledge:b1bde62ca7719f21fe42
  - 图谱路径：requirement:train_only_preprocessing → strategy:train_only_pipeline → Knowledge:b1bde62ca7719f21fe42
  - 图谱路径：requirement:unknown_categories → strategy:unknown_safe_encoding → Knowledge:b1bde62ca7719f21fe42
- 文本预处理与适用条件 — `text_business.md`:6（Knowledge:d0dcebab4212d8e42da3）
  - 图谱路径：requirement:missing_values → strategy:imputation → Knowledge:d0dcebab4212d8e42da3
  - 图谱路径：requirement:train_only_preprocessing → strategy:train_only_pipeline → Knowledge:d0dcebab4212d8e42da3
- 评价指标与业务约束 — `churn_business.md`:14（Knowledge:ee594e7236a1cbe8c520）
  - 图谱路径：requirement:class_imbalance → strategy:class_weighting → Knowledge:ee594e7236a1cbe8c520
- churn_category_encoder — `historical_code.py`:5（Function:abe0662caba9d48ee2eb）
  - 图谱路径：requirement:unknown_categories → strategy:unknown_safe_encoding → Function:abe0662caba9d48ee2eb
- 候选算法及适用条件 — `churn_business.md`:11（Knowledge:b8867b632089cbc5844c）
  - 图谱路径：requirement:class_imbalance → strategy:class_weighting → Knowledge:b8867b632089cbc5844c
- 历史失败案例：未知类别 — `churn_business.md`:20（Knowledge:948b7c74dfbb766ab40d）
  - 图谱路径：requirement:unknown_categories → strategy:unknown_safe_encoding → Knowledge:948b7c74dfbb766ab40d

历史经验检索数：0

## 候选比较（验证集）

| 算法 | 通过 | 尝试次数 | ROC-AUC | F1 |
|---|---|---|---|---|
| logistic | True | 1 | 0.8274521531100478 | 0.6470588235294118 |
| forest | True | 2 | 0.7963516746411483 | 0.6326530612244898 |

## 方案依据与修复轨迹

### logistic

逻辑回归作为线性基线模型，适合处理具有缺失值和类别不平衡的表格数据。结合标准化处理和 balanced 权重，可以有效捕捉主要特征与流失率之间的线性关系，提供良好的可解释性。

- attempt 0: passed=True, injected=False
  - code: `logistic/attempt_0/algorithm.py`; SHA256: `efb9b95257d91ea12245e7c29ccd29234c9ad04302cec3279dc937e2f611f47d`
  - checks: `{'probability_contract': True, 'estimator_matches_plan': True, 'hyperparameters_match_plan': True, 'preprocessing_matches_plan': True, 'missing_column_rejected': True, 'column_order': True, 'repeat_prediction': True, 'model_reload': True, 'unknown_category_contract': True, 'unknown_category_mixed_contract': True, 'unknown_category_payment_method': True, 'unknown_category_mixed_payment_method': True, 'unknown_category': True, 'all_missing_values': True, 'minimum_roc_auc': True, 'minimum_f1': True}`
  - explanation: The implementation creates a sklearn Pipeline with a ColumnTransformer for preprocessing numeric and categorical features separately. Numeric features are imputed with median and scaled. Categorical features are imputed with most frequent and encoded with OneHotEncoder that ignores unknown categories. The final classifier is LogisticRegression with balanced class weights, C=1.0, solver='lbfgs', max_iter=1000, and random_state set to 42. The predict function checks for missing required fields and returns the probability of the positive class (class 1).
### forest

随机森林能够捕捉特征间的非线性交互作用，适合在客户流失数据中寻找复杂的决策模式。通过限制树深度和叶子节点样本数，可以有效控制过拟合，并在类别不平衡情况下通过 balanced 权重提升模型性能。

- attempt 0: passed=False, injected=False
  - code: `forest/attempt_0/algorithm.py`; SHA256: `5398192201fc8662e955849e92bf3e264d37c11a8e68aca8ebedf76a0029e292`
  - checks: `{}`
  - explanation: The implementation constructs a scikit-learn Pipeline to build an end-to-end churn prediction model. It uses separate transformers for numeric and categorical features to handle preprocessing correctly. Numeric features are imputed with the median and scaled. Categorical features are imputed with the most frequent value and one-hot encoded, configured to ignore unknown categories. The final step is a RandomForestClassifier with the specified hyperparameters, including class_weight='balanced' to address the class imbalance. The `build_model` function returns an unfitted pipeline, and the `predict` function returns the probability of the positive class for a given DataFrame, raising a ValueError if required features are missing. The code uses only the allowed imports and follows the specified constraints.

```text
Traceback (most recent call last):
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\forge\worker.py", line 109, in <module>
    result = execute(Path(sys.argv[1]), Path(sys.argv[2]))
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\forge\worker.py", line 64, in execute
    task.check_preprocessing(model, expected["preprocessing"])
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\forge\tasks.py", line 71, in check_preprocessing
    raise ValueError(f"Plan mismatch: ColumnTransformer must explicitly group {names}")
ValueError: Plan mismatch: ColumnTransformer must explicitly group ['tenure', 'monthly_charges', 'support_calls']

```

- attempt 1: passed=True, injected=False
  - code: `forest/attempt_1/algorithm.py`; SHA256: `6825ee3f875ba1d4972b2094b15004528b4ce99d17dfad6a594165038653c608`
  - checks: `{'probability_contract': True, 'estimator_matches_plan': True, 'hyperparameters_match_plan': True, 'preprocessing_matches_plan': True, 'missing_column_rejected': True, 'column_order': True, 'repeat_prediction': True, 'model_reload': True, 'unknown_category_contract': True, 'unknown_category_mixed_contract': True, 'unknown_category_payment_method': True, 'unknown_category_mixed_payment_method': True, 'unknown_category': True, 'all_missing_values': True, 'minimum_roc_auc': True, 'minimum_f1': True}`
  - explanation: The previous code failed because it used a nested Pipeline for preprocessing instead of a ColumnTransformer. The validation error indicated that the ColumnTransformer must explicitly group the numeric feature names ['tenure', 'monthly_charges', 'support_calls']. The fix implements a ColumnTransformer that applies the numeric_transformer to the numeric_features and the categorical_transformer to the categorical_features. The rest of the implementation, including the RandomForestClassifier hyperparameters and the predict function, remains consistent with the validated plan.

## 最终测试集评估

选定算法：logistic

```json
{
  "checks": {
    "probability_contract": true,
    "estimator_matches_plan": true,
    "hyperparameters_match_plan": true,
    "preprocessing_matches_plan": true,
    "missing_column_rejected": true,
    "column_order": true,
    "repeat_prediction": true,
    "model_reload": true,
    "unknown_category_contract": true,
    "unknown_category_mixed_contract": true,
    "unknown_category_payment_method": true,
    "unknown_category_mixed_payment_method": true,
    "unknown_category": true,
    "all_missing_values": true,
    "minimum_roc_auc": true,
    "minimum_f1": true
  },
  "check_errors": {},
  "executed": true,
  "worker_seconds": 2.1232,
  "python_peak_mb": 50.905,
  "wall_seconds": 2.5638,
  "metrics": {
    "roc_auc": 0.8356991845363939,
    "f1": 0.7,
    "accuracy": 0.75
  },
  "passed": true
}
```

## 解释边界

数据为固定种子的模拟数据，指标不能代表真实客户业务效果。按验证集选择，最终测试结果不参与修复。
Mock 的需求解析、规划、代码生成和修复是规则/模板行为，不能作为真实大模型效果证据。
角色按状态机顺序交接；未实现自主并行多智能体，也未实现操作系统级安全沙箱。
python_peak_mb 仅统计 Python 内存分配，非进程总内存；wall_seconds 包含进程启动。
