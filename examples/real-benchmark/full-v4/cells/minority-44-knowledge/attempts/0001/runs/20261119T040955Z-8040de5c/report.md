# 算法能力验证报告

- Run: `20261119T040955Z-8040de5c`
- 状态：passed
- 模式：glm / glm-4.7-flash
- 故障注入：False
- 数据 SHA256：`a80959c53c0b16bd2c0347537dcd64f175d093ddc91106c6ba51dd4bf36a0100`

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
        "missing_rate": 0.063889,
        "cardinality": 71,
        "constant": false,
        "quantiles": {
          "0.25": 19.0,
          "0.5": 37.0,
          "0.75": 55.0
        }
      },
      "monthly_charges": {
        "kind": "numeric",
        "missing_rate": 0.047222,
        "cardinality": 343,
        "constant": false,
        "quantiles": {
          "0.25": 42.54325333506832,
          "0.5": 69.47080129149407,
          "0.75": 94.84542815373783
        }
      },
      "support_calls": {
        "kind": "numeric",
        "missing_rate": 0.083333,
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
        "missing_rate": 0.066667,
        "cardinality": 3,
        "constant": false
      },
      "payment_method": {
        "kind": "categorical",
        "missing_rate": 0.1,
        "cardinality": 2,
        "constant": false
      }
    },
    "positive_rate": 0.20277777777777778,
    "minority_rate": 0.20277777777777778,
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
        "rationale": "使用随机森林处理潜在的类别不平衡（正样本率 0.20）和非线性特征交互。根据数据概况，限制树深度以控制过拟合，并使用 balanced 权重。此模型适合作为复杂非线性关系的基线进行比较。",
        "hyperparameters": {
          "n_estimators": 100,
          "max_depth": 7,
          "min_samples_leaf": 5,
          "class_weight": "balanced"
        }
      },
      {
        "algorithm": "logistic",
        "rationale": "逻辑回归作为可解释的线性基线模型，配合标准化和 balanced 类别权重。它适合用于快速验证特征重要性并与随机森林的结果进行对比。",
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
    "evidence_ids": [
      "Knowledge:b1bde62ca7719f21fe42",
      "Knowledge:ee594e7236a1cbe8c520",
      "Knowledge:b8867b632089cbc5844c",
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
| forest | True | 2 | 0.7695312500000001 | 0.45454545454545453 |
| logistic | False | 3 | N/A | N/A |

## 方案依据与修复轨迹

### forest

使用随机森林处理潜在的类别不平衡（正样本率 0.20）和非线性特征交互。根据数据概况，限制树深度以控制过拟合，并使用 balanced 权重。此模型适合作为复杂非线性关系的基线进行比较。

- attempt 0: passed=False, injected=False
  - code: `forest/attempt_0/algorithm.py`; SHA256: `559b5f15ab4293d6f624bd13355e188f14d712ae9007bb35f2501dea92ca49c6`
  - checks: `{'static_check': False}`
  - explanation: 实现遵循了指定的 JSON 对象结构和约束条件。1. **模型构建 (`build_model`)**：返回一个 sklearn Pipeline，包含一个 ColumnTransformer 和一个 RandomForestClassifier。预处理器处理数值特征（中位数填补 -> 标准化）和类别特征（众数填补 -> 独热编码，设置 `handle_unknown='ignore'`）。分类器使用了指定的超参数（n_estimators=100, max_depth=7, min_samples_leaf=5, class_weight='balanced'），并设置了 random_state 和 n_jobs=1。2. **预测 (`predict`)**：接收模型和 DataFrame，首先检查必需的字段，如果缺失则抛出 ValueError。然后使用 `model.predict_proba` 并返回正类（索引 1）的概率。

```text
Prohibited name: __name__
```

- attempt 1: passed=True, injected=False
  - code: `forest/attempt_1/algorithm.py`; SHA256: `b202b8747e7cbdab4488e98c4ad424e9568a6cf3d563e09d4e9880455babe895`
  - checks: `{'probability_contract': True, 'estimator_matches_plan': True, 'hyperparameters_match_plan': True, 'preprocessing_matches_plan': True, 'missing_column_rejected': True, 'column_order': True, 'repeat_prediction': True, 'model_reload': True, 'unknown_category_contract': True, 'unknown_category_mixed_contract': True, 'unknown_category_payment_method': True, 'unknown_category_mixed_payment_method': True, 'unknown_category': True, 'all_missing_values': True, 'minimum_roc_auc': True, 'minimum_f1': True}`
  - explanation: The implementation constructs a sklearn Pipeline that first preprocesses numeric features (median imputation, standardization) and categorical features (most frequent imputation, one-hot encoding with unknown category handling). The final step is a RandomForestClassifier with balanced class weights to handle imbalance, set random_state for reproducibility, and n_jobs=1 as specified. The predict function validates input for required fields, raises ValueError for missing ones, and returns the positive class probabilities.
### logistic

逻辑回归作为可解释的线性基线模型，配合标准化和 balanced 类别权重。它适合用于快速验证特征重要性并与随机森林的结果进行对比。

- attempt 0: passed=False, injected=False
  - code: `logistic/attempt_0/algorithm.py`; SHA256: `ad3ae491785dd59fcb9348d1f569710b4650d889cbff558f1802a5a09bebb98a`
  - checks: `{}`
  - explanation: The implementation follows the specified requirements:

1.  **Pipeline Structure**: Uses a `sklearn.pipeline.Pipeline` where the final step is a custom `ProbabilityExtractor` wrapping the `LogisticRegression` classifier.
2.  **Preprocessing**:
    -   **Numeric**: Imputed with median and scaled using `StandardScaler`.
    -   **Categorical**: Imputed with most frequent and encoded using `OneHotEncoder` with `handle_unknown='ignore'` to satisfy the unknown categories constraint.
3.  **Classifier**: `LogisticRegression` is configured with `class_weight='balanced'` to address class imbalance, and `random_state` is set for reproducibility.
4.  **Interface**:
    -   `build_model(seed)` returns the unfitted pipeline.
    -   `predict(model, frame)` checks for required columns and returns 1D probabilities for class 1.
5.  **Constraints**: Only imports `numpy`, `pandas`, and `sklearn`. No filesystem or dynamic imports are used.

```text
Traceback (most recent call last):
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\forge\worker.py", line 109, in <module>
    result = execute(Path(sys.argv[1]), Path(sys.argv[2]))
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\forge\worker.py", line 56, in execute
    raise ValueError(f"Plan mismatch: expected classifier {expected_class}")
ValueError: Plan mismatch: expected classifier LogisticRegression

```

- attempt 1: passed=False, injected=False
  - code: `logistic/attempt_1/algorithm.py`; SHA256: `0cac3b4aeb01a1da699afe703f768eecf5188e8dcf3cc2f6f2741515d3dae7e9`
  - checks: `{}`
  - explanation: The implementation follows the specified requirements precisely. The pipeline uses a ColumnTransformer to apply appropriate preprocessing to numeric and categorical features: median imputation and scaling for numeric features, and most-frequent imputation with One-Hot Encoding that handles unknown categories for categorical features. The classifier is a Logistic Regression with balanced class weights, L2 regularization (C=1.0), and the lbfgs solver. The build_model function returns an unfitted pipeline, and the predict function validates input data for required features and returns the finite positive-class probabilities in the order of the input rows. The random_state is set to 42 to ensure reproducibility.

```text
Traceback (most recent call last):
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\forge\worker.py", line 109, in <module>
    result = execute(Path(sys.argv[1]), Path(sys.argv[2]))
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\forge\worker.py", line 63, in execute
    raise ValueError(f"Plan mismatch: {key} must equal {value!r}")
ValueError: Plan mismatch: random_state must equal 44

```

- attempt 2: passed=False, injected=False
  - code: `logistic/attempt_2/algorithm.py`; SHA256: `dd07d7a71f41335b4d9888c8c5b43b9fe8a53bfc9e03ae53c213a589319b2006`
  - checks: `{}`
  - explanation: The implementation follows the specified interface_contract:

1.  **Model Selection**: The 'active_hyperparameters' and 'candidate_contract' specify using `LogisticRegression` with class_weight='balanced'. This handles the 'class_imbalance' constraint mentioned in the plan and knowledge.

2.  **Preprocessing**: A `sklearn.Pipeline` is constructed using `ColumnTransformer`. Numeric features use `SimpleImputer(strategy='median')` followed by `StandardScaler`. Categorical features use `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(handle_unknown='ignore')`. This satisfies the 'missing_values', 'unknown_categories', and 'preprocessing' requirements.

3.  **Pipeline Structure**: The classifier is the last step in the pipeline, as required. The `build_model` function returns an unfitted pipeline.

4.  **Prediction**: The `predict` function checks for missing required fields and raises a `ValueError` if any are found. It uses the classifier's `predict_proba` to return the probability of class 1. It validates that the returned probabilities are finite.

5.  **Constraints**: The `build_model` function sets `random_state=seed` and `n_jobs=1` (though not a parameter for LogisticRegression, it's a general constraint for forests, but the code uses LogisticRegression). The model uses only numpy, pandas, and sklearn imports.

```text
Traceback (most recent call last):
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\forge\worker.py", line 109, in <module>
    result = execute(Path(sys.argv[1]), Path(sys.argv[2]))
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\forge\worker.py", line 68, in execute
    probabilities = vector(module.predict(model, evaluation.copy()), len(evaluation))
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\runtime\glm\benchmark-full-v4\cells\minority-44-knowledge\attempts\0001\runs\20261119T040955Z-8040de5c\logistic\attempt_2\algorithm.py", line 103, in predict
    probs = classifier.predict_proba(frame)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\work\.venv312\Lib\site-packages\sklearn\linear_model\_logistic.py", line 1600, in predict_proba
    return super()._predict_proba_lr(X)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\work\.venv312\Lib\site-packages\sklearn\linear_model\_base.py", line 437, in _predict_proba_lr
    prob = self.decision_function(X)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\work\.venv312\Lib\site-packages\sklearn\linear_model\_base.py", line 389, in decision_function
    X = validate_data(self, X, accept_sparse="csr", reset=False)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
ValueError: could not convert string to float: 'annual'

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
  "worker_seconds": 2.7789,
  "python_peak_mb": 53.577,
  "wall_seconds": 3.2052,
  "metrics": {
    "roc_auc": 0.7925347222222223,
    "f1": 0.47619047619047616,
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
