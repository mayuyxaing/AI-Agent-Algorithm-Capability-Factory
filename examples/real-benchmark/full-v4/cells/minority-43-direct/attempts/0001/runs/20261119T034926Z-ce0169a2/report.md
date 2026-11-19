# 算法能力验证报告

- Run: `20261119T034926Z-ce0169a2`
- 状态：no_valid_candidate
- 模式：glm / glm-4.7-flash
- 故障注入：False
- 数据 SHA256：`10ff7f6cf2641f927c954282bea4208e1611532510669eb5056503e1f3d0bafe`

## 需求

客户流失预测，处理缺失值、未知类别和类别不平衡，比较逻辑回归和随机森林，使用 ROC-AUC 和 F1 评价。

## 检索依据

知识抽取审计：status=passed，LLM 接受 0 条，拒绝 0 条；不合格事实未进入能力图谱。

工作流 v3；实验组=direct；每候选生成预算=3；未向真实模型提供完整算法模板。

## 训练数据画像与实现计划

```json
{
  "data_profile": {
    "scope": "training_only",
    "rows": 360,
    "columns": {
      "tenure": {
        "kind": "numeric",
        "missing_rate": 0.102778,
        "cardinality": 72,
        "constant": false,
        "quantiles": {
          "0.25": 18.0,
          "0.5": 36.0,
          "0.75": 55.0
        }
      },
      "monthly_charges": {
        "kind": "numeric",
        "missing_rate": 0.094444,
        "cardinality": 326,
        "constant": false,
        "quantiles": {
          "0.25": 49.50843907373198,
          "0.5": 71.75373354952998,
          "0.75": 96.08808946635546
        }
      },
      "support_calls": {
        "kind": "numeric",
        "missing_rate": 0.102778,
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
        "missing_rate": 0.072222,
        "cardinality": 3,
        "constant": false
      },
      "payment_method": {
        "kind": "categorical",
        "missing_rate": 0.077778,
        "cardinality": 2,
        "constant": false
      }
    },
    "positive_rate": 0.20555555555555555,
    "minority_rate": 0.20555555555555555,
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
        "rationale": "逻辑回归作为可解释的线性基线模型，适合处理类别不平衡问题（正类率约20.56%）。标准化数值特征有助于提升收敛速度和模型稳定性。使用类别权重'balanced'以补偿少数类样本不足。选择'lbfgs'求解器，因为它对稀疏数据和中小规模数据集表现良好。",
        "hyperparameters": {
          "C": 1.0,
          "max_iter": 1000,
          "class_weight": "balanced",
          "solver": "lbfgs"
        }
      },
      {
        "algorithm": "forest",
        "rationale": "随机森林能够捕捉特征间的非线性关系和交互作用，适合处理复杂的客户流失模式。通过设置较高的树深度（如8）和较大的最小叶子样本数（如5）以防止过拟合。类别权重'balanced'确保少数类得到足够关注。",
        "hyperparameters": {
          "n_estimators": 100,
          "max_depth": 8,
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
    "evidence_ids": [],
    "experience_ids": [],
    "experience_applied": false
  }
}
```


历史经验检索数：0

## 候选比较（验证集）

| 算法 | 通过 | 尝试次数 | ROC-AUC | F1 |
|---|---|---|---|---|
| logistic | False | 3 | N/A | N/A |
| forest | False | 3 | 0.7202105263157894 | 0.42857142857142855 |

## 方案依据与修复轨迹

### logistic

逻辑回归作为可解释的线性基线模型，适合处理类别不平衡问题（正类率约20.56%）。标准化数值特征有助于提升收敛速度和模型稳定性。使用类别权重'balanced'以补偿少数类样本不足。选择'lbfgs'求解器，因为它对稀疏数据和中小规模数据集表现良好。

- attempt 0: passed=False, injected=False
  - code: `logistic/attempt_0/algorithm.py`; SHA256: `84eeb8a7a1431eca49f8e598cff98a5d8861c7e90ba95d31c1998a978b2e3995`
  - checks: `{}`
  - explanation: The implementation constructs a scikit-learn Pipeline for churn classification using a Logistic Regression classifier. The pipeline consists of two main stages: preprocessing and classification. The preprocessing stage handles missing values, scales numeric features, and one-hot encodes categorical features. The classification stage uses a Logistic Regression model with balanced class weights to address the class imbalance. The 'build_model' function returns an unfitted pipeline, and the 'predict' function validates input features and returns positive class probabilities. The code uses only numpy, pandas, and scikit-learn imports, adhering to the specified constraints.

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

- attempt 1: passed=False, injected=False
  - code: `logistic/attempt_1/algorithm.py`; SHA256: `c5fca7b568408f59d62582538877a314d212dc5031f83ab949ba2ec0f9686f5e`
  - checks: `{'static_check': False}`
  - explanation: This implementation creates a churn classification pipeline using Logistic Regression as the classifier. The pipeline handles missing values with median imputation for numeric features and most-frequent imputation for categorical features. StandardScaler is used for numeric features, and OneHotEncoder with 'ignore' strategy handles unknown categories. The model uses balanced class weights to address the class imbalance (positive rate ~20.56%). The predict function validates input features and returns positive-class probabilities, raising errors for missing features or invalid probabilities.

```text
Import outside numpy/pandas/sklearn allowlist
```

- attempt 2: passed=False, injected=False
  - code: `logistic/attempt_2/algorithm.py`; SHA256: `9a92cb3078799d5fca6d85849b67109118189a330c6bcda875f960ff5f908c4f`
  - checks: `{}`
  - explanation: The solution implements the specified interface. `build_model` constructs a scikit-learn Pipeline with a preprocessing step (handling missing values, scaling, and one-hot encoding) and a LogisticRegression classifier as the final step, configured with the provided hyperparameters. `predict` validates the input frame for required features and returns the positive class probabilities from the fitted model, raising appropriate errors for missing fields or non-finite values.

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

### forest

随机森林能够捕捉特征间的非线性关系和交互作用，适合处理复杂的客户流失模式。通过设置较高的树深度（如8）和较大的最小叶子样本数（如5）以防止过拟合。类别权重'balanced'确保少数类得到足够关注。

- attempt 0: passed=False, injected=False
  - code: `forest/attempt_0/algorithm.py`; SHA256: `2e468ea3734d1175c48176a1835cb598600a6491c6cb7d90d52046c86bf63d1e`
  - checks: `{'probability_contract': True, 'estimator_matches_plan': True, 'hyperparameters_match_plan': True, 'preprocessing_matches_plan': True, 'missing_column_rejected': True, 'column_order': True, 'repeat_prediction': True, 'model_reload': True, 'unknown_category_contract': True, 'unknown_category_mixed_contract': True, 'unknown_category_payment_method': True, 'unknown_category_mixed_payment_method': True, 'unknown_category': True, 'all_missing_values': True, 'minimum_roc_auc': True, 'minimum_f1': False}`
  - explanation: The implementation follows the specified requirements:

1. **Model Building (`build_model`)**:
   - Creates a scikit-learn Pipeline with preprocessing steps followed by a RandomForestClassifier.
   - **Numeric Features**: Uses `SimpleImputer` with `strategy='median'` to handle missing values, followed by `StandardScaler` for normalization.
   - **Categorical Features**: Uses `SimpleImputer` with `strategy='most_frequent'` to handle missing values, followed by `OneHotEncoder` with `handle_unknown='ignore'` to encode categories and ignore unknown ones.
   - **RandomForestClassifier**: Configured with `n_estimators=100`, `max_depth=8`, `min_samples_leaf=5`, `class_weight='balanced'`, and `random_state=42` for reproducibility. `n_jobs=1` is set as per instructions.

2. **Prediction (`predict`)**:
   - Checks for the presence of all required features (`tenure`, `monthly_charges`, `support_calls`, `contract`, `payment_method`). Raises a `ValueError` if any are missing.
   - Uses the fitted model's `predict_proba` method to obtain class probabilities.
   - Extracts the probability of the positive class (index 1) for each row.
   - Validates that all probabilities are finite (no NaN or Inf values).
   - Returns the 1D array of positive-class probabilities in the same order as the input rows.

The implementation adheres to the constraints, uses only the allowed imports, and does not perform any module-level training or filesystem/network operations.

```text
{} Failed checks: ['minimum_f1']
```

- attempt 1: passed=False, injected=False
  - code: `forest/attempt_1/algorithm.py`; SHA256: `dd8cacf6f325d6cab9382fc47297fd0a15a84c9b923e05b2147e211f7f4a5743`
  - checks: `{'probability_contract': True, 'estimator_matches_plan': True, 'hyperparameters_match_plan': True, 'preprocessing_matches_plan': True, 'missing_column_rejected': True, 'column_order': True, 'repeat_prediction': True, 'model_reload': True, 'unknown_category_contract': True, 'unknown_category_mixed_contract': True, 'unknown_category_payment_method': True, 'unknown_category_mixed_payment_method': True, 'unknown_category': True, 'all_missing_values': True, 'minimum_roc_auc': True, 'minimum_f1': False}`
  - explanation: The implementation creates a scikit-learn Pipeline that handles preprocessing (imputation, scaling, encoding) and model training. The model is built using RandomForestClassifier with the specified hyperparameters. The `predict` function validates the input, ensures the presence of required features, and returns the probability of the positive class (churn).

```text
{} Failed checks: ['minimum_f1']
```

- attempt 2: passed=False, injected=False
  - code: `forest/attempt_2/algorithm.py`; SHA256: `7379358965d9359d055be173b5a3bdc107594539073ebca7f896290db215e100`
  - checks: `{'probability_contract': True, 'estimator_matches_plan': True, 'hyperparameters_match_plan': True, 'preprocessing_matches_plan': True, 'missing_column_rejected': True, 'column_order': True, 'repeat_prediction': True, 'model_reload': True, 'unknown_category_contract': True, 'unknown_category_mixed_contract': True, 'unknown_category_payment_method': True, 'unknown_category_mixed_payment_method': True, 'unknown_category': True, 'all_missing_values': True, 'minimum_roc_auc': True, 'minimum_f1': False}`
  - explanation: The implementation creates a scikit-learn Pipeline with a ColumnTransformer for preprocessing and a RandomForestClassifier as the final step. The preprocessing includes median imputation and standard scaling for numeric features, and most frequent imputation with one-hot encoding for categorical features. The classifier is configured with the specified hyperparameters and a random state for reproducibility. The predict function validates input data for required fields and returns the probability of the positive class.

```text
{} Failed checks: ['minimum_f1']
```


## 最终测试集评估

选定算法：None

```json
null
```

## 解释边界

数据为固定种子的模拟数据，指标不能代表真实客户业务效果。按验证集选择，最终测试结果不参与修复。
Mock 的需求解析、规划、代码生成和修复是规则/模板行为，不能作为真实大模型效果证据。
角色按状态机顺序交接；未实现自主并行多智能体，也未实现操作系统级安全沙箱。
python_peak_mb 仅统计 Python 内存分配，非进程总内存；wall_seconds 包含进程启动。
