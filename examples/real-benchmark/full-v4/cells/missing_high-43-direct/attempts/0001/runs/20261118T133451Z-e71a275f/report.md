# 算法能力验证报告

- Run: `20261118T133451Z-e71a275f`
- 状态：passed
- 模式：glm / glm-4.7-flash
- 故障注入：False
- 数据 SHA256：`e2b64050532ce82160943ac3e6cbb0504a4526df7c04b4cd4af2bc29ad0c796a`

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
        "missing_rate": 0.294444,
        "cardinality": 71,
        "constant": false,
        "quantiles": {
          "0.25": 21.0,
          "0.5": 39.0,
          "0.75": 58.0
        }
      },
      "monthly_charges": {
        "kind": "numeric",
        "missing_rate": 0.261111,
        "cardinality": 266,
        "constant": false,
        "quantiles": {
          "0.25": 45.00381034562399,
          "0.5": 64.5098455878981,
          "0.75": 95.61564553859307
        }
      },
      "support_calls": {
        "kind": "numeric",
        "missing_rate": 0.258333,
        "cardinality": 7,
        "constant": false,
        "quantiles": {
          "0.25": 0.5,
          "0.5": 1.0,
          "0.75": 2.0
        }
      },
      "contract": {
        "kind": "categorical",
        "missing_rate": 0.247222,
        "cardinality": 3,
        "constant": false
      },
      "payment_method": {
        "kind": "categorical",
        "missing_rate": 0.3,
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
        "algorithm": "forest",
        "rationale": "随机森林能有效捕捉 tenure、monthly_charges 和 support_calls 之间的非线性关系，并自动处理特征交互。对于 contract 和 payment_method 等类别特征，其鲁棒性优于线性模型。结合 class_weight='balanced' 处理 churn 类别不平衡。",
        "hyperparameters": {
          "n_estimators": 100,
          "max_depth": 7,
          "min_samples_leaf": 5,
          "class_weight": "balanced"
        }
      },
      {
        "algorithm": "logistic",
        "rationale": "逻辑回归提供可解释的线性基线模型。使用 class_weight='balanced' 解决 churn 类别不平衡问题。对数值特征进行标准化以改善收敛速度。",
        "hyperparameters": {
          "C": 1.0,
          "max_iter": 1000,
          "class_weight": "balanced",
          "solver": "lbfgs"
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
| forest | True | 1 | 0.6749401913875599 | 0.5360824742268041 |
| logistic | True | 2 | 0.7517942583732058 | 0.5794392523364486 |

## 方案依据与修复轨迹

### forest

随机森林能有效捕捉 tenure、monthly_charges 和 support_calls 之间的非线性关系，并自动处理特征交互。对于 contract 和 payment_method 等类别特征，其鲁棒性优于线性模型。结合 class_weight='balanced' 处理 churn 类别不平衡。

- attempt 0: passed=True, injected=False
  - code: `forest/attempt_0/algorithm.py`; SHA256: `aee4c0493150b3963f6e1e33881e6903701dd2552f14b24446c8b2ed80046fe0`
  - checks: `{'probability_contract': True, 'estimator_matches_plan': True, 'hyperparameters_match_plan': True, 'preprocessing_matches_plan': True, 'missing_column_rejected': True, 'column_order': True, 'repeat_prediction': True, 'model_reload': True, 'unknown_category_contract': True, 'unknown_category_mixed_contract': True, 'unknown_category_payment_method': True, 'unknown_category_mixed_payment_method': True, 'unknown_category': True, 'all_missing_values': True, 'minimum_roc_auc': True, 'minimum_f1': True}`
  - explanation: The implementation creates a RandomForestClassifier pipeline using sklearn. It handles missing values with median imputation for numeric features and most-frequent imputation for categorical features. Numeric features are scaled using StandardScaler. One-hot encoding is applied to categorical features with 'ignore' strategy for unknown categories. The classifier uses the specified hyperparameters (n_estimators=100, max_depth=7, min_samples_leaf=5, class_weight='balanced') to handle class imbalance. The predict function validates input data contains all required features and returns class 1 probabilities as a 1D numpy array.
### logistic

逻辑回归提供可解释的线性基线模型。使用 class_weight='balanced' 解决 churn 类别不平衡问题。对数值特征进行标准化以改善收敛速度。

- attempt 0: passed=False, injected=False
  - code: `logistic/attempt_0/algorithm.py`; SHA256: `3fd74d15f40667729c8b23e259f6d411d27b16a800c191fcb391aa9f23171cf6`
  - checks: `{}`
  - explanation: The solution implements a `sklearn.Pipeline` as requested, with `LogisticRegression` as the final classifier. It uses `SimpleImputer` with median for numeric features and most frequent for categorical features to handle missing values. It applies `StandardScaler` to numeric features and `OneHotEncoder` with `handle_unknown='ignore'` for categorical features to handle unknown categories. The classifier uses `class_weight='balanced'` to address class imbalance and `n_jobs=1` as specified. The `predict` function validates the presence of all required features and returns the probability of the positive class.

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
  - code: `logistic/attempt_1/algorithm.py`; SHA256: `04f82ce1df2b1b84b98cdc74ea3ba96336d970b8e5c3c86832c1189dfe7c22f0`
  - checks: `{'probability_contract': True, 'estimator_matches_plan': True, 'hyperparameters_match_plan': True, 'preprocessing_matches_plan': True, 'missing_column_rejected': True, 'column_order': True, 'repeat_prediction': True, 'model_reload': True, 'unknown_category_contract': True, 'unknown_category_mixed_contract': True, 'unknown_category_payment_method': True, 'unknown_category_mixed_payment_method': True, 'unknown_category': True, 'all_missing_values': True, 'minimum_roc_auc': True, 'minimum_f1': True}`
  - explanation: The implementation uses a scikit-learn Pipeline as requested. It defines the features, applies median imputation and scaling to numeric features, and most-frequent imputation with One-Hot Encoding to categorical features. The final step is the LogisticRegression classifier with the specified hyperparameters (C=1.0, max_iter=1000, class_weight='balanced', solver='lbfgs'). The build_model function returns this unfitted pipeline. The predict function checks for missing required fields, raises ValueError, and returns the positive class probabilities.

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
  "worker_seconds": 2.2569,
  "python_peak_mb": 50.888,
  "wall_seconds": 2.7381,
  "metrics": {
    "roc_auc": 0.7819976076555023,
    "f1": 0.6226415094339622,
    "accuracy": 0.6666666666666666
  },
  "passed": true
}
```

## 解释边界

数据为固定种子的模拟数据，指标不能代表真实客户业务效果。按验证集选择，最终测试结果不参与修复。
Mock 的需求解析、规划、代码生成和修复是规则/模板行为，不能作为真实大模型效果证据。
角色按状态机顺序交接；未实现自主并行多智能体，也未实现操作系统级安全沙箱。
python_peak_mb 仅统计 Python 内存分配，非进程总内存；wall_seconds 包含进程启动。
