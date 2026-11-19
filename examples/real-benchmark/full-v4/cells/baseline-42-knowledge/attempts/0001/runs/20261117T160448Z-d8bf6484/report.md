# 算法能力验证报告

- Run: `20261117T160448Z-d8bf6484`
- 状态：passed
- 模式：glm / glm-4.7-flash
- 故障注入：False
- 数据 SHA256：`636769745426526c68263c70abce8af51e58eda7ea0ea72616830dc65fb7c50e`

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
        "algorithm": "logistic",
        "rationale": "逻辑回归适合作为可解释的线性基线模型。鉴于数据集中特征较少且存在类别不平衡（正负样本率均为36.1%），使用 class_weight='balanced' 可以有效平衡损失函数。数据集中数值特征（tenure, monthly_charges, support_calls）的缺失率较低（约3%），可以使用中位数填补。类别特征（contract, payment_method）缺失率也较低，可以使用众数填补。根据业务约束，需要处理未知类别，因此类别编码器需设置 handle_unknown='ignore'。逻辑回归通常对特征的尺度敏感，因此需要对数值特征进行标准化。",
        "hyperparameters": {
          "C": 1.0,
          "max_iter": 1000,
          "class_weight": "balanced",
          "solver": "lbfgs"
        }
      },
      {
        "algorithm": "forest",
        "rationale": "随机森林适合捕捉特征间的非线性关系和交互作用，且对类别不平衡具有较好的鲁棒性（通过 class_weight='balanced'）。在处理未知类别时，需要使用独热编码并设置 handle_unknown='ignore'。数据集中数值特征的缺失率较低，可以使用中位数填补；类别特征使用众数填补。随机森林通常不需要对数值特征进行缩放，但为了保持一致性，也可以选择缩放。根据数据规模（360行），设置适当的树深度（max_depth）以防止过拟合。",
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
| logistic | False | 3 | N/A | N/A |
| forest | True | 1 | 0.7363334340078526 | 0.6095238095238096 |

## 方案依据与修复轨迹

### logistic

逻辑回归适合作为可解释的线性基线模型。鉴于数据集中特征较少且存在类别不平衡（正负样本率均为36.1%），使用 class_weight='balanced' 可以有效平衡损失函数。数据集中数值特征（tenure, monthly_charges, support_calls）的缺失率较低（约3%），可以使用中位数填补。类别特征（contract, payment_method）缺失率也较低，可以使用众数填补。根据业务约束，需要处理未知类别，因此类别编码器需设置 handle_unknown='ignore'。逻辑回归通常对特征的尺度敏感，因此需要对数值特征进行标准化。

- attempt 0: passed=False, injected=False
  - code: `logistic/attempt_0/algorithm.py`; SHA256: `0601866b89806c8bd5b704563ae02e2a2d802c3fb27fe3561f6837fc88b2c14a`
  - checks: `{}`
  - explanation: 该实现构建了一个用于客户流失预测的 Pipeline。预处理步骤首先对数值特征使用中位数进行填补，然后进行标准化，确保模型对特征尺度不敏感。对类别特征使用众数进行填补，并应用独热编码，同时设置 handle_unknown='ignore' 以处理未知类别。分类器选用逻辑回归，配置了 class_weight='balanced' 来应对类别不平衡问题，并设置了 max_iter=1000 以确保算法收敛。predict 函数会检查输入数据是否包含所有必需的特征，如果缺少则抛出 ValueError。最终返回正类的概率。

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
  - code: `logistic/attempt_1/algorithm.py`; SHA256: `d4a82a0b8a53a81814e551203d459168be58be831a22c822f97d4e521248fa92`
  - checks: `{}`
  - explanation: This implementation builds a logistic regression pipeline tailored for churn prediction. It follows the specified constraints:

1.  **Pipeline Structure**: A single `sklearn.pipeline.Pipeline` is used, with preprocessing steps (imputation, scaling, encoding) followed by the `LogisticRegression` classifier as the final step.
2.  **Preprocessing**:
    -   **Numeric Features** (`tenure`, `monthly_charges`, `support_calls`): Handled with a `SimpleImputer` (median) and `StandardScaler`.
    -   **Categorical Features** (`contract`, `payment_method`): Handled with a `SimpleImputer` (most frequent) and `OneHotEncoder` with `handle_unknown='ignore'` to manage unknown categories during prediction.
3.  **Classifier**: `LogisticRegression` is used with `class_weight='balanced'` to address the class imbalance, and `random_state=seed` for reproducibility. The hyperparameters `C=1.0`, `max_iter=1000`, and `solver='lbfgs'` are set as specified.
4.  **Model Construction**: The `build_model` function returns an **unfitted** pipeline, as required. It accepts a `seed` parameter and sets the classifier's random state to this seed.
5.  **Prediction**: The `predict` function takes a model and a frame. It performs input validation to ensure all required features are present, raising a `ValueError` if any are missing. It then uses the model's `predict_proba` method to obtain the probability of the positive class (churn) and returns the 1D array of these probabilities.

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

- attempt 2: passed=False, injected=False
  - code: `logistic/attempt_2/algorithm.py`; SHA256: `8d87772b9d6e91a1af451dbf4b2c5f3a8defee7fd93c3fdbe6718f545e9513f4`
  - checks: `{}`
  - explanation: This implementation follows the specified requirements. It uses a sklearn Pipeline with a Logistic Regression classifier as the final step. The pipeline handles preprocessing as required: numeric features are imputed with median and scaled, while categorical features are imputed with most frequent and encoded using OneHotEncoder with 'handle_unknown=ignore'. The classifier uses 'class_weight=balanced' to address the class imbalance and is seeded for reproducibility. The `predict` function validates input data for the presence of required fields and returns the probability of the positive class (class 1).

```text
Traceback (most recent call last):
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\forge\worker.py", line 109, in <module>
    result = execute(Path(sys.argv[1]), Path(sys.argv[2]))
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\forge\worker.py", line 41, in execute
    spec.loader.exec_module(module)
  File "<frozen importlib._bootstrap_external>", line 999, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\runtime\glm\benchmark-full-v4\cells\baseline-42-knowledge\attempts\0001\runs\20261117T160448Z-d8bf6484\logistic\attempt_2\algorithm.py", line 3, in <module>
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer
ImportError: cannot import name 'SimpleImputer' from 'sklearn.preprocessing' (C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\work\.venv312\Lib\site-packages\sklearn\preprocessing\__init__.py)

```

### forest

随机森林适合捕捉特征间的非线性关系和交互作用，且对类别不平衡具有较好的鲁棒性（通过 class_weight='balanced'）。在处理未知类别时，需要使用独热编码并设置 handle_unknown='ignore'。数据集中数值特征的缺失率较低，可以使用中位数填补；类别特征使用众数填补。随机森林通常不需要对数值特征进行缩放，但为了保持一致性，也可以选择缩放。根据数据规模（360行），设置适当的树深度（max_depth）以防止过拟合。

- attempt 0: passed=True, injected=False
  - code: `forest/attempt_0/algorithm.py`; SHA256: `97a34e9f99738fb7d1f2548f362711bd6b55f0f79a125bb715fda1d48b6d23dc`
  - checks: `{'probability_contract': True, 'estimator_matches_plan': True, 'hyperparameters_match_plan': True, 'preprocessing_matches_plan': True, 'missing_column_rejected': True, 'column_order': True, 'repeat_prediction': True, 'model_reload': True, 'unknown_category_contract': True, 'unknown_category_mixed_contract': True, 'unknown_category_payment_method': True, 'unknown_category_mixed_payment_method': True, 'unknown_category': True, 'all_missing_values': True, 'minimum_roc_auc': True, 'minimum_f1': True}`
  - explanation: The implementation follows the specified requirements:

1. **Model Selection**: A Random Forest classifier (`RandomForestClassifier`) is used as the final step in the pipeline, as specified by the `candidate_contract` and `active_hyperparameters`. The hyperparameters (`n_estimators=100`, `max_depth=7`, `min_samples_leaf=5`, `class_weight='balanced'`) are set to balance performance and prevent overfitting, especially given the class imbalance.

2. **Preprocessing**:
   - **Numeric Features**: Median imputation (`SimpleImputer(strategy='median')`) followed by standard scaling (`StandardScaler`) is applied to `tenure`, `monthly_charges`, and `support_calls`. This handles missing values and scales the features.
   - **Categorical Features**: Most frequent imputation (`SimpleImputer(strategy='most_frequent')`) followed by one-hot encoding (`OneHotEncoder(handle_unknown='ignore')`) is applied to `contract` and `payment_method`. The `handle_unknown='ignore'` parameter ensures the model can handle unknown categories during prediction.
   - The preprocessing steps are combined using `ColumnTransformer` to handle feature groups separately.

3. **Pipeline Structure**: The preprocessing steps and classifier are combined into a `Pipeline` object, ensuring a single, unified workflow for training and prediction.

4. **Prediction Function**:
   - The `predict` function validates that all required features are present in the input DataFrame. If any feature is missing, it raises a `ValueError`.
   - It uses the fitted pipeline's `predict_proba` method to obtain class probabilities.
   - The probability of the positive class (churn) is extracted and returned as a 1D array.
   - It checks for non-finite probabilities and raises a `ValueError` if any are found.

5. **Constraints Compliance**:
   - The model uses `class_weight='balanced'` to address class imbalance.
   - The pipeline ensures preprocessing is applied only during training (fit) and reused during prediction (transform) to prevent data leakage.
   - The `random_state` is set in the classifier for reproducibility.
   - The `n_jobs=1` parameter in the classifier ensures single-threaded execution.

This implementation meets all specified requirements and constraints.

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
  "worker_seconds": 2.7972,
  "python_peak_mb": 53.632,
  "wall_seconds": 3.365,
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
