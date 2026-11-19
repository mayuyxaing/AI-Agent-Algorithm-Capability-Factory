# 算法能力验证报告

- Run: `20261118T014536Z-ebac6333`
- 状态：passed
- 模式：glm / glm-4.7-flash
- 故障注入：False
- 数据 SHA256：`1d63afca3af17abd358cc8f28d2dbcc3a65fd007fa4a59b8ec0f23b131bb6422`

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
        "missing_rate": 0.277778,
        "cardinality": 66,
        "constant": false,
        "quantiles": {
          "0.25": 20.0,
          "0.5": 38.0,
          "0.75": 56.0
        }
      },
      "monthly_charges": {
        "kind": "numeric",
        "missing_rate": 0.241667,
        "cardinality": 273,
        "constant": false,
        "quantiles": {
          "0.25": 40.90576586022415,
          "0.5": 67.98634005510651,
          "0.75": 93.37534974379162
        }
      },
      "support_calls": {
        "kind": "numeric",
        "missing_rate": 0.269444,
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
        "missing_rate": 0.286111,
        "cardinality": 3,
        "constant": false
      },
      "payment_method": {
        "kind": "categorical",
        "missing_rate": 0.238889,
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
        "rationale": "作为可解释的线性基线模型，配合标准化处理数值特征，并使用class_weight='balanced'处理类别不平衡问题，以提供可靠的性能参考。",
        "hyperparameters": {
          "C": 1.0,
          "max_iter": 1000,
          "class_weight": "balanced",
          "solver": "lbfgs"
        }
      },
      {
        "algorithm": "forest",
        "rationale": "适合捕捉数据中的非线性关系和特征交互，限制树深度和叶节点样本数以控制过拟合，使用class_weight='balanced'处理类别不平衡。",
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
| forest | True | 1 | 0.754605859257022 | 0.6336633663366337 |

## 方案依据与修复轨迹

### logistic

作为可解释的线性基线模型，配合标准化处理数值特征，并使用class_weight='balanced'处理类别不平衡问题，以提供可靠的性能参考。

- attempt 0: passed=False, injected=False
  - code: `logistic/attempt_0/algorithm.py`; SHA256: `aab7fdf66ba2e5d74232bc426ff7e0bd8778a057339b4c80aa8504e346c5314e`
  - checks: `{}`
  - explanation: The implementation follows the specified requirements:
1. **Pipeline Structure**: A sklearn Pipeline is used with a preprocessor and the classifier as the final step.
2. **Preprocessing**:
   - Numeric features ('tenure', 'monthly_charges', 'support_calls') are imputed with the median and scaled using StandardScaler.
   - Categorical features ('contract', 'payment_method') are imputed with the most frequent value and encoded using OneHotEncoder with `handle_unknown='ignore'` to handle unknown categories.
3. **Classifier**: LogisticRegression is used with the provided hyperparameters (C=1.0, max_iter=1000, class_weight='balanced', solver='lbfgs') and `random_state=seed` for reproducibility.
4. **Model Fitting**: The model is returned unfitted to adhere to the requirement of not training at module import.
5. **Prediction**: The `predict` function checks for missing required features and raises a ValueError if any are absent. It returns the positive class probabilities as a 1D array.

```text
Traceback (most recent call last):
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\forge\worker.py", line 109, in <module>
    result = execute(Path(sys.argv[1]), Path(sys.argv[2]))
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\forge\worker.py", line 42, in execute
    model = module.build_model(seed=payload["seed"])
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\runtime\glm\benchmark-full-v4\cells\missing_high-42-knowledge\attempts\0002\runs\20261118T014536Z-ebac6333\logistic\attempt_0\algorithm.py", line 22, in build_model
    preprocessor = ColumnTransformer(
                   ^^^^^^^^^^^^^^^^^
NameError: name 'ColumnTransformer' is not defined

```

- attempt 1: passed=False, injected=False
  - code: `logistic/attempt_1/algorithm.py`; SHA256: `a0c917d368ee927c74aef65a81f1147133114358eb76a7b1a2cebc202b86ff4d`
  - checks: `{}`
  - explanation: The implementation builds a robust churn classification pipeline using a logistic regression model. It starts with a custom `FeatureExtractor` transformer that performs all specified preprocessing steps: median imputation for numeric features (`tenure`, `monthly_charges`, `support_calls`), most frequent imputation for categorical features (`contract`, `payment_method`), standard scaling for numeric features, and one-hot encoding for categorical features. The one-hot encoder is configured with `handle_unknown='ignore'` to handle potential unknown categories in production data. The final pipeline uses a `LogisticRegression` classifier with `class_weight='balanced'` to address class imbalance, `random_state` for reproducibility, and other specified hyperparameters. The `predict` function validates input data for required fields and returns the probability of the positive class (churn).

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
  - code: `logistic/attempt_2/algorithm.py`; SHA256: `04207a8ba61731ec7dea947cfd6fbfa2d56a7b9e311af2b414603733fa5502dc`
  - checks: `{}`
  - explanation: The implementation creates a sklearn Pipeline with a ColumnTransformer for preprocessing. Numeric features are imputed with the median and scaled. Categorical features are imputed with the most frequent value and encoded using OneHotEncoder with `handle_unknown='ignore'`. The final step is a ClassifierWrapper around a LogisticRegression estimator initialized with the specified hyperparameters. The `predict` function validates the presence of required columns and returns the probability of the positive class.

```text
Traceback (most recent call last):
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\forge\worker.py", line 109, in <module>
    result = execute(Path(sys.argv[1]), Path(sys.argv[2]))
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\forge\worker.py", line 41, in execute
    spec.loader.exec_module(module)
  File "<frozen importlib._bootstrap_external>", line 999, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\runtime\glm\benchmark-full-v4\cells\missing_high-42-knowledge\attempts\0002\runs\20261118T014536Z-ebac6333\logistic\attempt_2\algorithm.py", line 5, in <module>
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer
ImportError: cannot import name 'SimpleImputer' from 'sklearn.preprocessing' (C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\work\.venv312\Lib\site-packages\sklearn\preprocessing\__init__.py)

```

### forest

适合捕捉数据中的非线性关系和特征交互，限制树深度和叶节点样本数以控制过拟合，使用class_weight='balanced'处理类别不平衡。

- attempt 0: passed=True, injected=False
  - code: `forest/attempt_0/algorithm.py`; SHA256: `c997a92c9b364940bd44fbe93110e5503732e2c07df8142d5147ec8abc12b590`
  - checks: `{'probability_contract': True, 'estimator_matches_plan': True, 'hyperparameters_match_plan': True, 'preprocessing_matches_plan': True, 'missing_column_rejected': True, 'column_order': True, 'repeat_prediction': True, 'model_reload': True, 'unknown_category_contract': True, 'unknown_category_mixed_contract': True, 'unknown_category_payment_method': True, 'unknown_category_mixed_payment_method': True, 'unknown_category': True, 'all_missing_values': True, 'minimum_roc_auc': True, 'minimum_f1': True}`
  - explanation: The implementation follows the specified requirements and knowledge. It constructs a scikit-learn Pipeline that first preprocesses numeric features (median imputation, scaling) and categorical features (most frequent imputation, one-hot encoding with unknown handling). The final step is a RandomForestClassifier with balanced class weights, limited tree depth to prevent overfitting, and a fixed random state. The `predict` function validates input fields and returns finite probabilities for the positive class.

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
  "worker_seconds": 2.5749,
  "python_peak_mb": 53.554,
  "wall_seconds": 3.0089,
  "metrics": {
    "roc_auc": 0.7958320749018424,
    "f1": 0.62,
    "accuracy": 0.6833333333333333
  },
  "passed": true
}
```

## 解释边界

数据为固定种子的模拟数据，指标不能代表真实客户业务效果。按验证集选择，最终测试结果不参与修复。
Mock 的需求解析、规划、代码生成和修复是规则/模板行为，不能作为真实大模型效果证据。
角色按状态机顺序交接；未实现自主并行多智能体，也未实现操作系统级安全沙箱。
python_peak_mb 仅统计 Python 内存分配，非进程总内存；wall_seconds 包含进程启动。
