# 算法能力验证报告

- Run: `20261117T162619Z-c6ddcd1a`
- 状态：passed
- 模式：glm / glm-4.7-flash
- 故障注入：False
- 数据 SHA256：`c31e27e02ce27d254e986664e747b758729ce355c16d445a7f40d6f18bd1fd90`

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
        "missing_rate": 0.044444,
        "cardinality": 71,
        "constant": false,
        "quantiles": {
          "0.25": 20.0,
          "0.5": 38.0,
          "0.75": 56.0
        }
      },
      "monthly_charges": {
        "kind": "numeric",
        "missing_rate": 0.036111,
        "cardinality": 347,
        "constant": false,
        "quantiles": {
          "0.25": 44.68083397059184,
          "0.5": 64.53495945339455,
          "0.75": 95.35571257135604
        }
      },
      "support_calls": {
        "kind": "numeric",
        "missing_rate": 0.033333,
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
        "missing_rate": 0.030556,
        "cardinality": 3,
        "constant": false
      },
      "payment_method": {
        "kind": "categorical",
        "missing_rate": 0.05,
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
        "rationale": "使用逻辑回归作为可解释的线性基线，处理类别不平衡问题。",
        "hyperparameters": {
          "C": 1.0,
          "max_iter": 1000,
          "class_weight": "balanced",
          "solver": "lbfgs"
        }
      },
      {
        "algorithm": "forest",
        "rationale": "使用随机森林捕捉非线性关系和特征交互，处理类别不平衡问题。",
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
      "Knowledge:b8867b632089cbc5844c",
      "Knowledge:ee594e7236a1cbe8c520",
      "Function:abe0662caba9d48ee2eb",
      "Knowledge:948b7c74dfbb766ab40d"
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
| logistic | True | 1 | 0.7945574162679426 | 0.62 |
| forest | True | 1 | 0.7435705741626795 | 0.594059405940594 |

## 方案依据与修复轨迹

### logistic

使用逻辑回归作为可解释的线性基线，处理类别不平衡问题。

- attempt 0: passed=True, injected=False
  - code: `logistic/attempt_0/algorithm.py`; SHA256: `21662b22d32e5248f3cbd582d268722f8738ec0f94f42b74225b5490b628ef33`
  - checks: `{'probability_contract': True, 'estimator_matches_plan': True, 'hyperparameters_match_plan': True, 'preprocessing_matches_plan': True, 'missing_column_rejected': True, 'column_order': True, 'repeat_prediction': True, 'model_reload': True, 'unknown_category_contract': True, 'unknown_category_mixed_contract': True, 'unknown_category_payment_method': True, 'unknown_category_mixed_payment_method': True, 'unknown_category': True, 'all_missing_values': True, 'minimum_roc_auc': True, 'minimum_f1': True}`
  - explanation: The implementation uses a sklearn Pipeline to combine preprocessing and the LogisticRegression classifier. The preprocessing includes SimpleImputer for missing values (median for numeric, most frequent for categorical) and OneHotEncoder for categorical features with 'handle_unknown=ignore' to handle unseen categories. StandardScaler is applied to numeric features. LogisticRegression is configured with class_weight='balanced' to address the class imbalance, along with the specified hyperparameters (C=1.0, max_iter=1000, solver='lbfgs'). The 'build_model' function returns an unfitted pipeline, and 'predict' validates input features and returns the probability of the positive class.
### forest

使用随机森林捕捉非线性关系和特征交互，处理类别不平衡问题。

- attempt 0: passed=True, injected=False
  - code: `forest/attempt_0/algorithm.py`; SHA256: `f0fe372267c980015045ad0561317a87572ffadf878c1b73fdaeabf4239ef02a`
  - checks: `{'probability_contract': True, 'estimator_matches_plan': True, 'hyperparameters_match_plan': True, 'preprocessing_matches_plan': True, 'missing_column_rejected': True, 'column_order': True, 'repeat_prediction': True, 'model_reload': True, 'unknown_category_contract': True, 'unknown_category_mixed_contract': True, 'unknown_category_payment_method': True, 'unknown_category_mixed_payment_method': True, 'unknown_category': True, 'all_missing_values': True, 'minimum_roc_auc': True, 'minimum_f1': True}`
  - explanation: The solution implements a `build_model` function that creates a scikit-learn Pipeline with a Random Forest classifier as the final step. The pipeline is built to handle the specific requirements:

1.  **Preprocessing**: A `ColumnTransformer` is used to apply different transformations to numeric and categorical features. Numeric features are imputed with the median and scaled. Categorical features are imputed with the most frequent value and encoded using `OneHotEncoder` with `handle_unknown='ignore'` to handle potential unknown categories in production.

2.  **Classifier**: The `RandomForestClassifier` is configured with the hyperparameters from the provided plan (`n_estimators=100`, `max_depth=7`, `min_samples_leaf=5`, `class_weight='balanced'`). It is seeded with `random_state=42` and set to `n_jobs=1` as specified.

3.  **Prediction**: The `predict` function validates that all required features are present in the input frame. It then uses the fitted pipeline's `predict_proba` method to return the probabilities for the positive class (class 1). If a required feature is missing, it raises a `ValueError`.

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
  "worker_seconds": 2.2742,
  "python_peak_mb": 50.905,
  "wall_seconds": 2.7036,
  "metrics": {
    "roc_auc": 0.7912679425837321,
    "f1": 0.6326530612244898,
    "accuracy": 0.7
  },
  "passed": true
}
```

## 解释边界

数据为固定种子的模拟数据，指标不能代表真实客户业务效果。按验证集选择，最终测试结果不参与修复。
Mock 的需求解析、规划、代码生成和修复是规则/模板行为，不能作为真实大模型效果证据。
角色按状态机顺序交接；未实现自主并行多智能体，也未实现操作系统级安全沙箱。
python_peak_mb 仅统计 Python 内存分配，非进程总内存；wall_seconds 包含进程启动。
