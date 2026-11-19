# 算法能力验证报告

- Run: `20261117T161736Z-60f218f1`
- 状态：passed
- 模式：glm / glm-4.7-flash
- 故障注入：False
- 数据 SHA256：`c31e27e02ce27d254e986664e747b758729ce355c16d445a7f40d6f18bd1fd90`

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
        "rationale": "逻辑回归作为线性基线模型，适合处理具有类别不平衡和少量缺失值的结构化数据。鉴于数据集中类别不平衡（正类率约36.4%），使用'balanced'类权重可有效平衡预测。对于数值特征，标准化（scale_numeric=true）有助于提升模型收敛速度和性能。缺失值通过中位数填充（numeric_imputer=median），分类特征通过众数填充（categorical_imputer=most_frequent）。未检测到未知类别，故设置为ignore。",
        "hyperparameters": {
          "C": 1.0,
          "max_iter": 1000,
          "class_weight": "balanced",
          "solver": "lbfgs"
        }
      },
      {
        "algorithm": "forest",
        "rationale": "随机森林能捕捉数据中的非线性关系和特征交互，适合处理客户流失预测中的复杂数据模式。基于数据特征（如tenure和monthly_charges的分布），设置适当的树深度（max_depth=7）和最小叶子样本数（min_samples_leaf=5）可防止过拟合。类别不平衡通过'balanced'权重处理。未检测到未知类别，故设置为ignore。",
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
| logistic | True | 2 | 0.7945574162679426 | 0.62 |
| forest | True | 3 | 0.7435705741626795 | 0.594059405940594 |

## 方案依据与修复轨迹

### logistic

逻辑回归作为线性基线模型，适合处理具有类别不平衡和少量缺失值的结构化数据。鉴于数据集中类别不平衡（正类率约36.4%），使用'balanced'类权重可有效平衡预测。对于数值特征，标准化（scale_numeric=true）有助于提升模型收敛速度和性能。缺失值通过中位数填充（numeric_imputer=median），分类特征通过众数填充（categorical_imputer=most_frequent）。未检测到未知类别，故设置为ignore。

- attempt 0: passed=False, injected=False
  - code: `logistic/attempt_0/algorithm.py`; SHA256: `e4b34910cad74b9833ef7957d35e441a511d5625b9f7c2965d45b88248656838`
  - checks: `{}`
  - explanation: The implementation follows the specified requirements and plan:

1. **Preprocessing**: 
   - Numeric features are handled with median imputation and standard scaling
   - Categorical features use most frequent imputation
   - Unknown categories are ignored (not handled separately)
   - Class imbalance is addressed using 'balanced' class weights

2. **Model**: 
   - LogisticRegression with the specified hyperparameters
   - Pipeline structure with preprocessing and classifier as final step
   - Random state is set for reproducibility
   - Single job for parallel processing

3. **Interface**: 
   - build_model returns an unfitted Pipeline
   - predict validates required features and returns 1D positive-class probabilities
   - Missing fields raise ValueError as specified

The implementation uses only numpy, pandas, and sklearn imports as required.

```text
Traceback (most recent call last):
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\forge\worker.py", line 109, in <module>
    result = execute(Path(sys.argv[1]), Path(sys.argv[2]))
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\forge\worker.py", line 67, in execute
    model.fit(train[FEATURES], train[task.target])
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\work\.venv312\Lib\site-packages\sklearn\base.py", line 1403, in wrapper
    return fit_method(estimator, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\work\.venv312\Lib\site-packages\sklearn\pipeline.py", line 645, in fit
    self._final_estimator.fit(Xt, y, **last_step_params["fit"])
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\work\.venv312\Lib\site-packages\sklearn\base.py", line 1403, in wrapper
    return fit_method(estimator, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\work\.venv312\Lib\site-packages\sklearn\linear_model\_logistic.py", line 1437, in fit
    X, y = validate_data(
           ^^^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\work\.venv312\Lib\site-packages\sklearn\utils\validation.py", line 3058, in validate_data
    X, y = check_X_y(X, y, **check_params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\work\.venv312\Lib\site-packages\sklearn\utils\validation.py", line 1330, in check_X_y
    X = check_array(
        ^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\work\.venv312\Lib\site-packages\sklearn\utils\validation.py", line 1038, in check_array
    array = _asarray_with_order(array, order=order, dtype=dtype, xp=xp)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\work\.venv312\Lib\site-packages\sklearn\utils\_array_api.py", line 983, in _asarray_with_order
    array = numpy.asarray(array, order=order, dtype=dtype)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: could not convert string to float: 'monthly'

```

- attempt 1: passed=True, injected=False
  - code: `logistic/attempt_1/algorithm.py`; SHA256: `fdeb9e5629e00c4a6d9f038e5c4f3fbfcd413e233628c550de94913850aa3c1c`
  - checks: `{'probability_contract': True, 'estimator_matches_plan': True, 'hyperparameters_match_plan': True, 'preprocessing_matches_plan': True, 'missing_column_rejected': True, 'column_order': True, 'repeat_prediction': True, 'model_reload': True, 'unknown_category_contract': True, 'unknown_category_mixed_contract': True, 'unknown_category_payment_method': True, 'unknown_category_mixed_payment_method': True, 'unknown_category': True, 'all_missing_values': True, 'minimum_roc_auc': True, 'minimum_f1': True}`
  - explanation: The implementation creates a scikit-learn Pipeline with a ColumnTransformer for preprocessing and LogisticRegression as the final estimator. Numeric features are imputed with median and scaled. Categorical features are imputed with most frequent and one-hot encoded with unknown categories ignored. The LogisticRegression uses the specified hyperparameters and 'balanced' class weights. The predict function validates the input DataFrame, checks for missing required columns, and returns the positive class probabilities.
### forest

随机森林能捕捉数据中的非线性关系和特征交互，适合处理客户流失预测中的复杂数据模式。基于数据特征（如tenure和monthly_charges的分布），设置适当的树深度（max_depth=7）和最小叶子样本数（min_samples_leaf=5）可防止过拟合。类别不平衡通过'balanced'权重处理。未检测到未知类别，故设置为ignore。

- attempt 0: passed=False, injected=False
  - code: `forest/attempt_0/algorithm.py`; SHA256: `a67d0ea8e35bb951a387d273130f5fca0eedce0e99b2d4aa05e548cde515c48d`
  - checks: `{}`
  - explanation: The implementation creates a sklearn Pipeline that first preprocesses numeric features using median imputation and categorical features using most-frequent imputation followed by one-hot encoding (ignoring unknown categories). The final step is a RandomForestClassifier with the specified hyperparameters (balanced class weights, max_depth=7, min_samples_leaf=5, n_estimators=100) and random_state=42. The predict function validates that all required features are present in the input frame and returns only the positive class probabilities from predict_proba, ensuring they are finite values.

```text
Traceback (most recent call last):
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\forge\worker.py", line 109, in <module>
    result = execute(Path(sys.argv[1]), Path(sys.argv[2]))
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\forge\worker.py", line 64, in execute
    task.check_preprocessing(model, expected["preprocessing"])
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\forge\tasks.py", line 76, in check_preprocessing
    raise ValueError("Plan mismatch: numeric scaling differs from plan")
ValueError: Plan mismatch: numeric scaling differs from plan

```

- attempt 1: passed=False, injected=False
  - code: `forest/attempt_1/algorithm.py`; SHA256: `a7fe07cdf785a8340182136c98bff1c0cb8cc2408aa012496017c088bc702cb7`
  - checks: `{}`
  - explanation: The implementation creates a sklearn Pipeline with a preprocessing step (ColumnTransformer) and a RandomForestClassifier. Preprocessing imputes missing numeric values with median and categorical values with most frequent, and scales numeric features. The classifier uses the specified hyperparameters (n_estimators=100, max_depth=7, min_samples_leaf=5, class_weight='balanced') and is seeded with random_state=42. The predict function validates input features and returns the probability of the positive class.

```text
Traceback (most recent call last):
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\forge\worker.py", line 109, in <module>
    result = execute(Path(sys.argv[1]), Path(sys.argv[2]))
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\forge\worker.py", line 67, in execute
    model.fit(train[FEATURES], train[task.target])
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\work\.venv312\Lib\site-packages\sklearn\base.py", line 1403, in wrapper
    return fit_method(estimator, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\work\.venv312\Lib\site-packages\sklearn\pipeline.py", line 645, in fit
    self._final_estimator.fit(Xt, y, **last_step_params["fit"])
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\work\.venv312\Lib\site-packages\sklearn\base.py", line 1403, in wrapper
    return fit_method(estimator, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\work\.venv312\Lib\site-packages\sklearn\ensemble\_forest.py", line 334, in fit
    X, y = validate_data(
           ^^^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\work\.venv312\Lib\site-packages\sklearn\utils\validation.py", line 3058, in validate_data
    X, y = check_X_y(X, y, **check_params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\work\.venv312\Lib\site-packages\sklearn\utils\validation.py", line 1330, in check_X_y
    X = check_array(
        ^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\work\.venv312\Lib\site-packages\sklearn\utils\validation.py", line 1038, in check_array
    array = _asarray_with_order(array, order=order, dtype=dtype, xp=xp)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\work\.venv312\Lib\site-packages\sklearn\utils\_array_api.py", line 983, in _asarray_with_order
    array = numpy.asarray(array, order=order, dtype=dtype)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: could not convert string to float: 'monthly'

```

- attempt 2: passed=True, injected=False
  - code: `forest/attempt_2/algorithm.py`; SHA256: `3af6b53d3de59ccbc2fd4c2658840a1f9fe37ae19886c4b0fe189b26c82cf398`
  - checks: `{'probability_contract': True, 'estimator_matches_plan': True, 'hyperparameters_match_plan': True, 'preprocessing_matches_plan': True, 'missing_column_rejected': True, 'column_order': True, 'repeat_prediction': True, 'model_reload': True, 'unknown_category_contract': True, 'unknown_category_mixed_contract': True, 'unknown_category_payment_method': True, 'unknown_category_mixed_payment_method': True, 'unknown_category': True, 'all_missing_values': True, 'minimum_roc_auc': True, 'minimum_f1': True}`
  - explanation: The solution implements a RandomForestClassifier pipeline using sklearn. It handles missing values with median imputation for numeric features and most frequent imputation for categorical features, scales numeric features using StandardScaler, and one-hot encodes categorical features. The classifier uses balanced class weights to address class imbalance and is configured with the specified hyperparameters (n_estimators=100, max_depth=7, min_samples_leaf=5). The predict function validates required fields and returns positive-class probabilities as a 1D array.

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
  "worker_seconds": 2.2596,
  "python_peak_mb": 51.11,
  "wall_seconds": 2.6866,
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
