# 算法能力验证报告

- Run: `20261119T033240Z-f727cb72`
- 状态：passed
- 模式：glm / glm-4.7-flash
- 故障注入：False
- 数据 SHA256：`baddc943c5358946bd163c5e846319aaa6dcde9f13a96b66fa9c85c8e163fc23`

## 需求

客户流失预测，处理缺失值、未知类别和类别不平衡，比较逻辑回归和随机森林，使用 ROC-AUC 和 F1 评价。

## 检索依据

知识抽取审计：status=passed，LLM 接受 0 条，拒绝 0 条；不合格事实未进入能力图谱。

工作流 v3；实验组=knowledge_repair；每候选生成预算=3；未向真实模型提供完整算法模板。

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
        "algorithm": "forest",
        "rationale": "随机森林擅长捕捉非线性关系和特征交互，适合处理客户流失数据中的复杂模式。",
        "hyperparameters": {
          "n_estimators": 100,
          "max_depth": 7,
          "min_samples_leaf": 5,
          "class_weight": "balanced"
        }
      },
      {
        "algorithm": "logistic",
        "rationale": "逻辑回归提供可解释的线性基线，结合标准化和类别权重处理不平衡。",
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
      "Function:abe0662caba9d48ee2eb"
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
| forest | True | 2 | 0.7238337320574162 | 0.5894736842105263 |
| logistic | False | 3 | N/A | N/A |

## 方案依据与修复轨迹

### forest

随机森林擅长捕捉非线性关系和特征交互，适合处理客户流失数据中的复杂模式。

- attempt 0: passed=False, injected=False
  - code: `forest/attempt_0/algorithm.py`; SHA256: `64bc3fe0059c5b51612fbb7d20f007f42aab89944e300e729072ddf9d9170b2a`
  - checks: `{}`
  - explanation: The implementation follows the specified contract using a Random Forest classifier within a sklearn Pipeline. The preprocessing handles missing values (median for numeric, most frequent for categorical), scales numeric features, and uses OneHotEncoder with `handle_unknown='ignore'` to manage unknown categories. The classifier uses the provided hyperparameters and a fixed random state. The `predict` function validates input presence and returns class-1 probabilities.

```text
Traceback (most recent call last):
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\forge\worker.py", line 109, in <module>
    result = execute(Path(sys.argv[1]), Path(sys.argv[2]))
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\forge\worker.py", line 41, in execute
    spec.loader.exec_module(module)
  File "<frozen importlib._bootstrap_external>", line 999, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\runtime\glm\benchmark-full-v4\cells\missing_high-44-knowledge_repair\attempts\0001\runs\20261119T033240Z-f727cb72\forest\attempt_0\algorithm.py", line 5, in <module>
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer
ImportError: cannot import name 'SimpleImputer' from 'sklearn.preprocessing' (C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\work\.venv312\Lib\site-packages\sklearn\preprocessing\__init__.py)

```

- attempt 1: passed=True, injected=False
  - code: `forest/attempt_1/algorithm.py`; SHA256: `11934297e4b8d4cecd40761cb8dd7c53c1e9990154a1fd04095e704e8fb18b37`
  - checks: `{'probability_contract': True, 'estimator_matches_plan': True, 'hyperparameters_match_plan': True, 'preprocessing_matches_plan': True, 'missing_column_rejected': True, 'column_order': True, 'repeat_prediction': True, 'model_reload': True, 'unknown_category_contract': True, 'unknown_category_mixed_contract': True, 'unknown_category_payment_method': True, 'unknown_category_mixed_payment_method': True, 'unknown_category': True, 'all_missing_values': True, 'minimum_roc_auc': True, 'minimum_f1': True}`
  - explanation: The code fixes the ImportError by removing the duplicate import of `SimpleImputer` and the unused `BaseEstimator, TransformerMixin`. The structure and logic remain consistent with the original implementation, adhering to the specified constraints and hyperparameters.
### logistic

逻辑回归提供可解释的线性基线，结合标准化和类别权重处理不平衡。

- attempt 0: passed=False, injected=False
  - code: `logistic/attempt_0/algorithm.py`; SHA256: `4571652f396516868769ae1c8e18549484062d1ef9b6437d7e441adce95dc7e7`
  - checks: `{}`
  - explanation: The implementation creates a pipeline with the specified preprocessing steps and LogisticRegression classifier. It handles missing values using median imputation for numeric features and most frequent imputation for categorical features. Categorical features are one-hot encoded with handle_unknown='ignore' to prevent errors on unseen categories. The classifier uses class_weight='balanced' to address class imbalance and includes the specified hyperparameters. The predict function validates input features and returns the probability of the positive class.

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
  - code: `logistic/attempt_1/algorithm.py`; SHA256: `4571652f396516868769ae1c8e18549484062d1ef9b6437d7e441adce95dc7e7`
  - checks: `{}`
  - explanation: The code implements a pipeline for churn prediction using Logistic Regression. It correctly applies median imputation for numeric features and most frequent imputation for categorical features, followed by scaling for numeric data and one-hot encoding for categorical data with 'ignore' strategy for unknown categories. The LogisticRegression uses class_weight='balanced', C=1.0, max_iter=1000, and solver='lbfgs' as specified. The predict function validates the presence of all required features and returns the probability of the positive class. The preprocessing steps are encapsulated within the pipeline to prevent data leakage.

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
  - code: `logistic/attempt_2/algorithm.py`; SHA256: `4571652f396516868769ae1c8e18549484062d1ef9b6437d7e441adce95dc7e7`
  - checks: `{}`
  - explanation: The code implements a pipeline for churn prediction using Logistic Regression. It correctly applies median imputation for numeric features and most frequent imputation for categorical features, followed by scaling for numeric data and one-hot encoding for categorical data with 'ignore' strategy for unknown categories. The LogisticRegression uses class_weight='balanced', C=1.0, max_iter=1000, and solver='lbfgs' as specified. The predict function validates the presence of all required features and returns the probability of the positive class. The preprocessing steps are encapsulated within the pipeline to prevent data leakage.

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
  "worker_seconds": 2.8719,
  "python_peak_mb": 53.563,
  "wall_seconds": 3.359,
  "metrics": {
    "roc_auc": 0.7671398369072788,
    "f1": 0.6736842105263158,
    "accuracy": 0.7416666666666667
  },
  "passed": true
}
```

## 解释边界

数据为固定种子的模拟数据，指标不能代表真实客户业务效果。按验证集选择，最终测试结果不参与修复。
Mock 的需求解析、规划、代码生成和修复是规则/模板行为，不能作为真实大模型效果证据。
角色按状态机顺序交接；未实现自主并行多智能体，也未实现操作系统级安全沙箱。
python_peak_mb 仅统计 Python 内存分配，非进程总内存；wall_seconds 包含进程启动。
