# 算法能力验证报告

- Run: `20261119T035203Z-4b859649`
- 状态：passed
- 模式：glm / glm-4.7-flash
- 故障注入：False
- 数据 SHA256：`10ff7f6cf2641f927c954282bea4208e1611532510669eb5056503e1f3d0bafe`

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
        "rationale": "使用逻辑回归作为线性基线，结合标准化处理数值特征和类别权重平衡少数类，利用 F1 和 ROC-AUC 评估性能。",
        "hyperparameters": {
          "C": 1.0,
          "max_iter": 1000,
          "class_weight": "balanced",
          "solver": "lbfgs"
        }
      },
      {
        "algorithm": "forest",
        "rationale": "使用随机森林捕捉非线性关系和特征交互，通过限制树深度和叶子节点样本数防止过拟合，同时利用类别权重处理不平衡数据。",
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
      "Knowledge:b8867b632089cbc5844c"
    ],
    "experience_ids": [],
    "experience_applied": true
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
| logistic | True | 1 | 0.7903157894736843 | 0.5396825396825397 |
| forest | False | 3 | 0.7193684210526315 | 0.43333333333333335 |

## 方案依据与修复轨迹

### logistic

使用逻辑回归作为线性基线，结合标准化处理数值特征和类别权重平衡少数类，利用 F1 和 ROC-AUC 评估性能。

- attempt 0: passed=True, injected=False
  - code: `logistic/attempt_0/algorithm.py`; SHA256: `d907f62bc85e86d78e80e991669aeaacb93075246584320d734ca4610f2f1445`
  - checks: `{'probability_contract': True, 'estimator_matches_plan': True, 'hyperparameters_match_plan': True, 'preprocessing_matches_plan': True, 'missing_column_rejected': True, 'column_order': True, 'repeat_prediction': True, 'model_reload': True, 'unknown_category_contract': True, 'unknown_category_mixed_contract': True, 'unknown_category_payment_method': True, 'unknown_category_mixed_payment_method': True, 'unknown_category': True, 'all_missing_values': True, 'minimum_roc_auc': True, 'minimum_f1': True}`
  - explanation: The implementation creates a scikit-learn Pipeline that first preprocesses the data using a ColumnTransformer. Numeric features are imputed with the median and scaled. Categorical features are imputed with the most frequent value and one-hot encoded, with `handle_unknown='ignore'` to handle unknown categories in production. The final step is a LogisticRegression classifier with the specified hyperparameters, including `class_weight='balanced'` to address class imbalance and `random_state` for reproducibility. The `predict` function validates the input for required features and returns the probability of the positive class.
### forest

使用随机森林捕捉非线性关系和特征交互，通过限制树深度和叶子节点样本数防止过拟合，同时利用类别权重处理不平衡数据。

- attempt 0: passed=False, injected=False
  - code: `forest/attempt_0/algorithm.py`; SHA256: `cce82fc224bcdd04ee15f0c8200f83158f5b149b91fc850d102dbe9d96c25e43`
  - checks: `{}`
  - explanation: The implementation follows the specified requirements:

1. **Pipeline Structure**: Uses a sklearn Pipeline with the classifier as the final step.

2. **Preprocessing**:
   - Numeric features (tenure, monthly_charges, support_calls) are imputed with median values and then scaled using StandardScaler.
   - Categorical features (contract, payment_method) are imputed with most frequent values and encoded using OneHotEncoder with handle_unknown='ignore' to handle unknown categories.

3. **Classifier**: RandomForestClassifier is configured with the specified hyperparameters (n_estimators=100, max_depth=7, min_samples_leaf=5, class_weight='balanced', n_jobs=1) and seeded with random_state=42.

4. **Feature Validation**: A custom transformer 'RequiredFeaturesCheck' is added at the start of the pipeline to validate that all required features are present, raising a ValueError if any are missing.

5. **Prediction**: The predict function extracts the positive class probabilities from the model's predict_proba output, ensuring they are finite and in the correct format (1D array of positive class probabilities).

```text
Traceback (most recent call last):
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\forge\worker.py", line 109, in <module>
    result = execute(Path(sys.argv[1]), Path(sys.argv[2]))
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\forge\worker.py", line 41, in execute
    spec.loader.exec_module(module)
  File "<frozen importlib._bootstrap_external>", line 999, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\runtime\glm\benchmark-full-v4\cells\minority-43-knowledge\attempts\0001\runs\20261119T035203Z-4b859649\forest\attempt_0\algorithm.py", line 5, in <module>
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer
ImportError: cannot import name 'SimpleImputer' from 'sklearn.preprocessing' (C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\work\.venv312\Lib\site-packages\sklearn\preprocessing\__init__.py)

```

- attempt 1: passed=False, injected=False
  - code: `forest/attempt_1/algorithm.py`; SHA256: `a5c242486e7f904be60e9921058d7976b21b0c0b74e4dc4077205f8e4e6540a5`
  - checks: `{}`
  - explanation: The implementation uses a scikit-learn Pipeline to ensure a fixed preprocessing flow. Numeric features ('tenure', 'monthly_charges', 'support_calls') are imputed with the median and scaled. Categorical features ('contract', 'payment_method') are imputed with the most frequent value and encoded using One-Hot Encoding with `handle_unknown='ignore'` to support unknown categories at prediction time. The classifier is a RandomForestClassifier configured with `class_weight='balanced'` to address the class imbalance, `n_jobs=1` for resource efficiency, and specific hyperparameters (`max_depth=7`, `min_samples_leaf=5`, `n_estimators=100`) as specified. The `predict` function validates the input for missing required fields and returns the probability of the positive class.

```text
Traceback (most recent call last):
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\forge\worker.py", line 109, in <module>
    result = execute(Path(sys.argv[1]), Path(sys.argv[2]))
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\forge\worker.py", line 68, in execute
    probabilities = vector(module.predict(model, evaluation.copy()), len(evaluation))
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\runtime\glm\benchmark-full-v4\cells\minority-43-knowledge\attempts\0001\runs\20261119T035203Z-4b859649\forest\attempt_1\algorithm.py", line 81, in predict
    probabilities = classifier.predict_proba(frame)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\work\.venv312\Lib\site-packages\sklearn\ensemble\_forest.py", line 945, in predict_proba
    X = self._validate_X_predict(X)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\work\.venv312\Lib\site-packages\sklearn\ensemble\_forest.py", line 615, in _validate_X_predict
    X = validate_data(
        ^^^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\work\.venv312\Lib\site-packages\sklearn\utils\validation.py", line 3041, in validate_data
    out = check_array(X, input_name="X", **check_params)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\work\.venv312\Lib\site-packages\sklearn\utils\validation.py", line 1038, in check_array
    array = _asarray_with_order(array, order=order, dtype=dtype, xp=xp)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\work\.venv312\Lib\site-packages\sklearn\utils\_array_api.py", line 983, in _asarray_with_order
    array = numpy.asarray(array, order=order, dtype=dtype)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\work\.venv312\Lib\site-packages\pandas\core\generic.py", line 1985, in __array__
    arr = np.asarray(values, dtype=dtype)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: could not convert string to float: 'monthly'

```

- attempt 2: passed=False, injected=False
  - code: `forest/attempt_2/algorithm.py`; SHA256: `a54d2605e5ed950f00f32799d3fae624f8188a0a0f4cc4ad479bdd7c2414e99e`
  - checks: `{'probability_contract': True, 'estimator_matches_plan': True, 'hyperparameters_match_plan': True, 'preprocessing_matches_plan': True, 'missing_column_rejected': True, 'column_order': True, 'repeat_prediction': True, 'model_reload': True, 'unknown_category_contract': True, 'unknown_category_mixed_contract': True, 'unknown_category_payment_method': True, 'unknown_category_mixed_payment_method': True, 'unknown_category': True, 'all_missing_values': True, 'minimum_roc_auc': True, 'minimum_f1': False}`
  - explanation: The implementation creates a sklearn Pipeline with a preprocessor and a Random Forest classifier. The preprocessor handles missing values using median for numeric features and most frequent for categorical features, and applies standard scaling to numeric features. It also performs one-hot encoding on categorical features with `handle_unknown='ignore'` to handle unseen categories. The classifier uses the specified hyperparameters (`n_estimators=100`, `max_depth=7`, `min_samples_leaf=5`, `class_weight='balanced'`) and a fixed random state. The `predict` function validates the input for required fields and returns the probability of the positive class.

```text
{} Failed checks: ['minimum_f1']
```


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
  "worker_seconds": 2.2436,
  "python_peak_mb": 51.108,
  "wall_seconds": 2.6614,
  "metrics": {
    "roc_auc": 0.8336842105263158,
    "f1": 0.5538461538461539,
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
