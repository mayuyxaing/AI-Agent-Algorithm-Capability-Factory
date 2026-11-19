# 算法能力验证报告

- Run: `20261119T033006Z-a31e470b`
- 状态：passed
- 模式：glm / glm-4.7-flash
- 故障注入：False
- 数据 SHA256：`baddc943c5358946bd163c5e846319aaa6dcde9f13a96b66fa9c85c8e163fc23`

## 需求

客户流失预测，处理缺失值、未知类别和类别不平衡，比较逻辑回归和随机森林，使用 ROC-AUC 和 F1 评价。

## 检索依据

知识抽取审计：status=passed，LLM 接受 0 条，拒绝 0 条；不合格事实未进入能力图谱。

工作流 v3；实验组=knowledge；每候选生成预算=3；未向真实模型提供完整算法模板。

## 训练数据画像与实现计划

```json
{
  "data_profile": {
    "scope": "training_only",
    "rows": 360,
    "columns": {
      "tenure": {
        "kind": "numeric",
        "missing_rate": 0.211111,
        "cardinality": 70,
        "constant": false,
        "quantiles": {
          "0.25": 19.75,
          "0.5": 34.5,
          "0.75": 55.25
        }
      },
      "monthly_charges": {
        "kind": "numeric",
        "missing_rate": 0.241667,
        "cardinality": 273,
        "constant": false,
        "quantiles": {
          "0.25": 41.27054562451961,
          "0.5": 69.47080129149407,
          "0.75": 96.89783701320889
        }
      },
      "support_calls": {
        "kind": "numeric",
        "missing_rate": 0.247222,
        "cardinality": 7,
        "constant": false,
        "quantiles": {
          "0.25": 1.0,
          "0.5": 1.0,
          "0.75": 3.0
        }
      },
      "contract": {
        "kind": "categorical",
        "missing_rate": 0.222222,
        "cardinality": 3,
        "constant": false
      },
      "payment_method": {
        "kind": "categorical",
        "missing_rate": 0.275,
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
        "rationale": "使用训练集的中位数填补缺失值并进行标准化，类别特征使用众数填补并做独热编码。逻辑回归作为可解释的线性基线，配合 class_weight='balanced' 处理类别不平衡，并使用 lbfgs 求解器。此预处理策略符合约束要求。",
        "hyperparameters": {
          "C": 1.0,
          "max_iter": 1000,
          "class_weight": "balanced",
          "solver": "lbfgs"
        }
      },
      {
        "algorithm": "forest",
        "rationale": "使用与逻辑回归相同的预处理流程。随机森林能够捕捉非线性关系和特征交互，限制树深度和叶节点样本数以控制过拟合。使用 class_weight='balanced' 处理类别不平衡。此预处理策略符合约束要求。",
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
      "Knowledge:b8867b632089cbc5844c"
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
| logistic | True | 1 | 0.7289174641148326 | 0.5904761904761905 |
| forest | True | 2 | 0.7238337320574162 | 0.5894736842105263 |

## 方案依据与修复轨迹

### logistic

使用训练集的中位数填补缺失值并进行标准化，类别特征使用众数填补并做独热编码。逻辑回归作为可解释的线性基线，配合 class_weight='balanced' 处理类别不平衡，并使用 lbfgs 求解器。此预处理策略符合约束要求。

- attempt 0: passed=True, injected=False
  - code: `logistic/attempt_0/algorithm.py`; SHA256: `2a5244f6927ec2d960dbc8b1844345ed1ae4be17ff0fc75863b8acc371dd85da`
  - checks: `{'probability_contract': True, 'estimator_matches_plan': True, 'hyperparameters_match_plan': True, 'preprocessing_matches_plan': True, 'missing_column_rejected': True, 'column_order': True, 'repeat_prediction': True, 'model_reload': True, 'unknown_category_contract': True, 'unknown_category_mixed_contract': True, 'unknown_category_payment_method': True, 'unknown_category_mixed_payment_method': True, 'unknown_category': True, 'all_missing_values': True, 'minimum_roc_auc': True, 'minimum_f1': True}`
  - explanation: The implementation follows the specified requirements:
1. **Pipeline Structure**: Uses sklearn Pipeline with preprocessor and classifier steps.
2. **Preprocessing**: Applies median imputation for numeric features, standard scaling, and most_frequent imputation with OneHotEncoder (handle_unknown='ignore') for categorical features.
3. **Classifier**: LogisticRegression with balanced class weights and lbfgs solver as specified in the candidate contract.
4. **Features**: Handles exactly the required features: tenure, monthly_charges, support_calls (numeric) and contract, payment_method (categorical).
5. **Prediction**: Returns positive class probabilities (1D array) and raises ValueError for missing fields.
6. **Constraints**: Satisfies all constraints including handling missing values, unknown categories, and class imbalance through the preprocessing and classifier settings.
### forest

使用与逻辑回归相同的预处理流程。随机森林能够捕捉非线性关系和特征交互，限制树深度和叶节点样本数以控制过拟合。使用 class_weight='balanced' 处理类别不平衡。此预处理策略符合约束要求。

- attempt 0: passed=False, injected=False
  - code: `forest/attempt_0/algorithm.py`; SHA256: `4498d1f90a2be7261e86cbacbe3e66a906e04bc2a905cfc851d423c6efb036cc`
  - checks: `{}`
  - explanation: The implementation builds a scikit-learn Pipeline that first preprocesses the data using a median imputer and scaler for numeric features, and a most frequent imputer and one-hot encoder for categorical features. The OneHotEncoder is configured with `handle_unknown='ignore'` to handle unseen categories. The final step of the pipeline is a RandomForestClassifier with `class_weight='balanced'` and specific hyperparameters (`n_estimators=100`, `max_depth=7`, `min_samples_leaf=5`) to manage overfitting. The `predict` function validates the input DataFrame for missing required fields and ensures the output probabilities are finite.

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
  - code: `forest/attempt_1/algorithm.py`; SHA256: `3ecd7e6bc473f1e8cb51c2c7823091693e144657ac8c03c90929271e3093a09e`
  - checks: `{'probability_contract': True, 'estimator_matches_plan': True, 'hyperparameters_match_plan': True, 'preprocessing_matches_plan': True, 'missing_column_rejected': True, 'column_order': True, 'repeat_prediction': True, 'model_reload': True, 'unknown_category_contract': True, 'unknown_category_mixed_contract': True, 'unknown_category_payment_method': True, 'unknown_category_mixed_payment_method': True, 'unknown_category': True, 'all_missing_values': True, 'minimum_roc_auc': True, 'minimum_f1': True}`
  - explanation: The implementation creates a scikit-learn Pipeline to handle the specified churn classification task. It includes a custom transformer to enforce the presence of required features, raising a ValueError if they are missing. The preprocessing stage uses a ColumnTransformer to apply median imputation and scaling to numeric features, and most-frequent imputation with One-Hot Encoding (with `handle_unknown='ignore'`) to categorical features. The final classifier is a RandomForestClassifier with the specified hyperparameters and `random_state` for reproducibility. The `predict` function extracts the positive-class probabilities (index 1) and validates the input and output.

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
  "worker_seconds": 2.2987,
  "python_peak_mb": 52.538,
  "wall_seconds": 2.7249,
  "metrics": {
    "roc_auc": 0.7774086378737541,
    "f1": 0.6666666666666666,
    "accuracy": 0.725
  },
  "passed": true
}
```

## 解释边界

数据为固定种子的模拟数据，指标不能代表真实客户业务效果。按验证集选择，最终测试结果不参与修复。
Mock 的需求解析、规划、代码生成和修复是规则/模板行为，不能作为真实大模型效果证据。
角色按状态机顺序交接；未实现自主并行多智能体，也未实现操作系统级安全沙箱。
python_peak_mb 仅统计 Python 内存分配，非进程总内存；wall_seconds 包含进程启动。
