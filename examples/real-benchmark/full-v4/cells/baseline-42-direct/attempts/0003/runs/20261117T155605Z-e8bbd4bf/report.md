# 算法能力验证报告

- Run: `20261117T155605Z-e8bbd4bf`
- 状态：passed
- 模式：glm / glm-4.7-flash
- 故障注入：False
- 数据 SHA256：`636769745426526c68263c70abce8af51e58eda7ea0ea72616830dc65fb7c50e`

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
        "missing_rate": 0.033333,
        "cardinality": 71,
        "constant": false,
        "quantiles": {
          "0.25": 20.0,
          "0.5": 37.0,
          "0.75": 56.0
        }
      },
      "monthly_charges": {
        "kind": "numeric",
        "missing_rate": 0.030556,
        "cardinality": 349,
        "constant": false,
        "quantiles": {
          "0.25": 40.197903768878156,
          "0.5": 67.78772740237457,
          "0.75": 94.23179494179767
        }
      },
      "support_calls": {
        "kind": "numeric",
        "missing_rate": 0.036111,
        "cardinality": 8,
        "constant": false,
        "quantiles": {
          "0.25": 1.0,
          "0.5": 1.0,
          "0.75": 2.0
        }
      },
      "contract": {
        "kind": "categorical",
        "missing_rate": 0.033333,
        "cardinality": 3,
        "constant": false
      },
      "payment_method": {
        "kind": "categorical",
        "missing_rate": 0.038889,
        "cardinality": 2,
        "constant": false
      }
    },
    "positive_rate": 0.3611111111111111,
    "minority_rate": 0.3611111111111111,
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
        "rationale": "随机森林能够捕捉特征间的非线性关系，适合处理如合同类型和支付方式等潜在交互效应，且对缺失值稳健。相比逻辑回归，它通常在处理不平衡数据时具有更高的ROC-AUC表现。",
        "hyperparameters": {
          "n_estimators": 100,
          "max_depth": 7,
          "min_samples_leaf": 5,
          "class_weight": "balanced"
        }
      },
      {
        "algorithm": "logistic",
        "rationale": "逻辑回归作为可解释的线性基线模型，适合快速评估特征重要性。通过标准化数值特征和平衡类别权重，它能在保持模型透明度的同时有效预测流失。",
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
| forest | True | 3 | 0.7363334340078526 | 0.6095238095238096 |
| logistic | False | 3 | N/A | N/A |

## 方案依据与修复轨迹

### forest

随机森林能够捕捉特征间的非线性关系，适合处理如合同类型和支付方式等潜在交互效应，且对缺失值稳健。相比逻辑回归，它通常在处理不平衡数据时具有更高的ROC-AUC表现。

- attempt 0: passed=False, injected=False
  - code: `forest/attempt_0/algorithm.py`; SHA256: `cf464063f17de00adad52f66db64040187de69fcc1ef1039c81292f77326f3a9`
  - checks: `{}`
  - explanation: The implementation uses a sklearn Pipeline with a ColumnTransformer for preprocessing and a RandomForestClassifier as the final estimator. Numeric features are imputed with the median and scaled, while categorical features are imputed with the most frequent value. The classifier is configured with the specified hyperparameters (n_estimators=100, max_depth=7, min_samples_leaf=5, class_weight='balanced') and random_state=42. The predict function ensures all required features are present before prediction and returns the probability of the positive class (class 1).

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

- attempt 1: passed=False, injected=False
  - code: `forest/attempt_1/algorithm.py`; SHA256: `c1242f370a75ff75a1c7dfb80ba2010cc397de280910246af4d5023bf4320190`
  - checks: `{}`
  - explanation: The implementation creates a scikit-learn Pipeline. It uses SimpleImputer for both numeric and categorical features with specified strategies ('median' and 'most_frequent'). Categorical features are encoded using OneHotEncoder with 'ignore' to handle unknown categories. Numeric features are scaled using StandardScaler. The final step is a RandomForestClassifier with the provided hyperparameters, set to 'balanced' class weight and a fixed random state. The predict function validates input frame columns and returns the positive class probabilities.

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

- attempt 2: passed=True, injected=False
  - code: `forest/attempt_2/algorithm.py`; SHA256: `c5bdbf28573b5b82cd7531a625c15f009d516b053287fa686715537139edc528`
  - checks: `{'probability_contract': True, 'estimator_matches_plan': True, 'hyperparameters_match_plan': True, 'preprocessing_matches_plan': True, 'missing_column_rejected': True, 'column_order': True, 'repeat_prediction': True, 'model_reload': True, 'unknown_category_contract': True, 'unknown_category_mixed_contract': True, 'unknown_category_payment_method': True, 'unknown_category_mixed_payment_method': True, 'unknown_category': True, 'all_missing_values': True, 'minimum_roc_auc': True, 'minimum_f1': True}`
  - explanation: The implementation follows the specified requirements:
1. Uses a sklearn Pipeline with preprocessing and classifier as the final step
2. Implements RandomForestClassifier with the specified hyperparameters
3. Handles missing values with median imputation for numeric features and most frequent imputation for categorical features
4. Scales numeric features using StandardScaler
5. Encodes categorical features using OneHotEncoder with 'ignore' strategy for unknown categories
6. Implements the interface_contract requirements:
   - build_model returns an unfitted Pipeline
   - predict returns finite 1D positive-class probabilities
   - predict raises ValueError for missing required fields
7. Uses only numpy, pandas, and sklearn imports
8. Sets random_state for reproducibility and n_jobs=1 as specified
### logistic

逻辑回归作为可解释的线性基线模型，适合快速评估特征重要性。通过标准化数值特征和平衡类别权重，它能在保持模型透明度的同时有效预测流失。

- attempt 0: passed=False, injected=False
  - code: `logistic/attempt_0/algorithm.py`; SHA256: `a5db3ab328b00352b7037c79dfdfaabe727ecbc2379af75e4da03e2aa5adce71`
  - checks: `{}`
  - explanation: The implementation builds a scikit-learn Pipeline with a preprocessing step (ColumnTransformer) and a LogisticRegression classifier as the final step. The preprocessing handles numeric features with median imputation and categorical features with most frequent imputation and one-hot encoding (with 'ignore' for unknown categories). The LogisticRegression is configured with the specified hyperparameters, including a balanced class weight and a fixed random state. The `predict` function validates that all required features are present in the input DataFrame before making predictions, returning the probability of the positive class.

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
  - code: `logistic/attempt_1/algorithm.py`; SHA256: `d753bd2efac23557b125ceb82dc6e64d552a32dca66697b80d10c15c74cf7443`
  - checks: `{}`
  - explanation: The implementation creates a scikit-learn Pipeline with a preprocessing step and a Logistic Regression classifier. The preprocessing step includes median imputation and standard scaling for numeric features, and most frequent imputation and one-hot encoding for categorical features. The classifier is configured with the specified hyperparameters, including balanced class weights and a fixed random state. The `predict` function validates the input DataFrame for required features, raises a ValueError if any are missing, and returns the probability of the positive class for each input row.

```text
Traceback (most recent call last):
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\forge\worker.py", line 109, in <module>
    result = execute(Path(sys.argv[1]), Path(sys.argv[2]))
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\forge\worker.py", line 42, in execute
    model = module.build_model(seed=payload["seed"])
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\runtime\glm\benchmark-full-v4\cells\baseline-42-direct\attempts\0003\runs\20261117T155605Z-e8bbd4bf\logistic\attempt_1\algorithm.py", line 29, in build_model
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: OneHotEncoder.__init__() got an unexpected keyword argument 'sparse'

```

- attempt 2: passed=False, injected=False
  - code: `logistic/attempt_2/algorithm.py`; SHA256: `024bbc3eea720a8b6ace7f42a801e5bd552d42e8d0050a3fe38bf0e50b154682`
  - checks: `{}`
  - explanation: The implementation follows the specified requirements:

1. **Model Selection**: Uses LogisticRegression as the classifier, with the exact hyperparameters C=1.0, max_iter=1000, class_weight='balanced', and solver='lbfgs' as specified in the plan.

2. **Preprocessing**: 
   - Numeric features (tenure, monthly_charges, support_calls) are imputed with median values and scaled using StandardScaler
   - Categorical features (contract, payment_method) are imputed with most_frequent values and encoded using OneHotEncoder with handle_unknown='ignore'

3. **Pipeline Structure**: Uses sklearn Pipeline with preprocessing steps followed by the classifier as the final step.

4. **Random State**: Set classifier random_state=seed (default 42) for reproducibility.

5. **Interface Contract**: 
   - build_model returns an unfitted Pipeline
   - predict checks for required fields and raises ValueError if missing, then returns class-1 probabilities
   - Uses only numpy, pandas, and sklearn imports as required
   - Does not perform any training at module import time

```text
Traceback (most recent call last):
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\forge\worker.py", line 109, in <module>
    result = execute(Path(sys.argv[1]), Path(sys.argv[2]))
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\forge\worker.py", line 42, in execute
    model = module.build_model(seed=payload["seed"])
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\runtime\glm\benchmark-full-v4\cells\baseline-42-direct\attempts\0003\runs\20261117T155605Z-e8bbd4bf\logistic\attempt_2\algorithm.py", line 23, in build_model
    preprocessor = ColumnTransformer(
                   ^^^^^^^^^^^^^^^^^
NameError: name 'ColumnTransformer' is not defined

```


## 最终测试集评估

选定算法：forest

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
  "worker_seconds": 2.8002,
  "python_peak_mb": 53.636,
  "wall_seconds": 3.3449,
  "metrics": {
    "roc_auc": 0.8574448807006947,
    "f1": 0.7184466019417476,
    "accuracy": 0.7583333333333333
  },
  "passed": true
}
```

## 解释边界

数据为固定种子的模拟数据，指标不能代表真实客户业务效果。按验证集选择，最终测试结果不参与修复。
Mock 的需求解析、规划、代码生成和修复是规则/模板行为，不能作为真实大模型效果证据。
角色按状态机顺序交接；未实现自主并行多智能体，也未实现操作系统级安全沙箱。
python_peak_mb 仅统计 Python 内存分配，非进程总内存；wall_seconds 包含进程启动。
