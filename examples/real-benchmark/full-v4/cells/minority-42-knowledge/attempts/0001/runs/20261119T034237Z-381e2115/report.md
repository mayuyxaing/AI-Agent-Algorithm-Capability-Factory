# 算法能力验证报告

- Run: `20261119T034237Z-381e2115`
- 状态：passed
- 模式：glm / glm-4.7-flash
- 故障注入：False
- 数据 SHA256：`5bac8d4d229c12bfae963c6bb6603450050beee8492e10acac43708b99b05390`

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
        "missing_rate": 0.075,
        "cardinality": 70,
        "constant": false,
        "quantiles": {
          "0.25": 21.0,
          "0.5": 40.0,
          "0.75": 57.0
        }
      },
      "monthly_charges": {
        "kind": "numeric",
        "missing_rate": 0.072222,
        "cardinality": 334,
        "constant": false,
        "quantiles": {
          "0.25": 43.264052663258184,
          "0.5": 67.58373015551754,
          "0.75": 93.34184988268841
        }
      },
      "support_calls": {
        "kind": "numeric",
        "missing_rate": 0.088889,
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
        "missing_rate": 0.102778,
        "cardinality": 3,
        "constant": false
      },
      "payment_method": {
        "kind": "categorical",
        "missing_rate": 0.075,
        "cardinality": 2,
        "constant": false
      }
    },
    "positive_rate": 0.2222222222222222,
    "minority_rate": 0.2222222222222222,
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
        "rationale": "作为线性基线模型，逻辑回归提供可解释性。鉴于数据集中存在类别不平衡（少数类占比约22.2%），必须设置 class_weight='balanced'。数值特征使用中位数填补后进行标准化，以符合算法输入要求。",
        "hyperparameters": {
          "C": 1.0,
          "max_iter": 1000,
          "class_weight": "balanced",
          "solver": "lbfgs"
        }
      },
      {
        "algorithm": "forest",
        "rationale": "随机森林能够捕捉特征间的非线性关系和交互作用，适合比较基线性能。为防止过拟合并控制计算资源，需限制树深度（max_depth）和叶节点最小样本数（min_samples_leaf）。同时处理类别不平衡。",
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
| logistic | False | 3 | N/A | N/A |
| forest | True | 1 | 0.712863401035444 | 0.47368421052631576 |

## 方案依据与修复轨迹

### logistic

作为线性基线模型，逻辑回归提供可解释性。鉴于数据集中存在类别不平衡（少数类占比约22.2%），必须设置 class_weight='balanced'。数值特征使用中位数填补后进行标准化，以符合算法输入要求。

- attempt 0: passed=False, injected=False
  - code: `logistic/attempt_0/algorithm.py`; SHA256: `8468b5da5d14a445468b55c494a51d2c0c304c9b48614f73a673683aa76016ed`
  - checks: `{}`
  - explanation: The implementation follows the specified requirements and preprocessing plan. It uses a sklearn Pipeline with a ColumnTransformer to handle numeric and categorical features separately. Numeric features are imputed with the median and scaled. Categorical features are imputed with the most frequent value and encoded using OneHotEncoder with `handle_unknown='ignore'` to manage potential unknown categories. The final step is a LogisticRegression classifier with `class_weight='balanced'` to address class imbalance. The `predict` function validates the presence of all required features and returns the probabilities of the positive class. The `build_model` function constructs this pipeline and returns it in an unfitted state.

```text
Traceback (most recent call last):
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\forge\worker.py", line 109, in <module>
    result = execute(Path(sys.argv[1]), Path(sys.argv[2]))
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\forge\worker.py", line 42, in execute
    model = module.build_model(seed=payload["seed"])
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\T-bags\Documents\Codex\2026-11-09\llm-agent-ai-agent-ai-agent\outputs\capability-forge\runtime\glm\benchmark-full-v4\cells\minority-42-knowledge\attempts\0001\runs\20261119T034237Z-381e2115\logistic\attempt_0\algorithm.py", line 41, in build_model
    preprocessor = ColumnTransformer(
                   ^^^^^^^^^^^^^^^^^
NameError: name 'ColumnTransformer' is not defined

```

- attempt 1: passed=False, injected=False
  - code: `logistic/attempt_1/algorithm.py`; SHA256: `4e14f7737f5bf32987511c0f2f79a5a51ab998d545c15ad31fed93aae90e470b`
  - checks: `{}`
  - explanation: The implementation creates a scikit-learn Pipeline with a preprocessing step and a LogisticRegression classifier. Numeric features are imputed with the median and scaled. Categorical features are imputed with the most frequent value and encoded using OneHotEncoder with `handle_unknown='ignore'` to handle potential new categories in prediction data. The Logistic Regression is configured with `class_weight='balanced'` to address the class imbalance. The `predict` function checks for required columns and returns the probability of the positive class (churn).

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
  - code: `logistic/attempt_2/algorithm.py`; SHA256: `23d0f132589691b4ac2742632fdc1293d0924dda1935f35a4eda0ec1a20e2d1f`
  - checks: `{}`
  - explanation: The solution implements a logistic regression pipeline for churn classification, adhering to the specified preprocessing and interface contract. The pipeline first handles numeric features by imputing missing values with the median and then standardizing them. Categorical features are imputed with the most frequent value and encoded using one-hot encoding, which is configured to 'ignore' unknown categories to prevent errors during prediction. The final classifier is a LogisticRegression with class_weight='balanced' to address the class imbalance and a fixed random_state for reproducibility. The `predict` function validates the presence of all required features and returns the 1D array of probabilities for the positive class.

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

随机森林能够捕捉特征间的非线性关系和交互作用，适合比较基线性能。为防止过拟合并控制计算资源，需限制树深度（max_depth）和叶节点最小样本数（min_samples_leaf）。同时处理类别不平衡。

- attempt 0: passed=True, injected=False
  - code: `forest/attempt_0/algorithm.py`; SHA256: `96d8e60ed428bc6933f5dbfab89bddfb5602d5415827eb66838f5416cdb1b873`
  - checks: `{'probability_contract': True, 'estimator_matches_plan': True, 'hyperparameters_match_plan': True, 'preprocessing_matches_plan': True, 'missing_column_rejected': True, 'column_order': True, 'repeat_prediction': True, 'model_reload': True, 'unknown_category_contract': True, 'unknown_category_mixed_contract': True, 'unknown_category_payment_method': True, 'unknown_category_mixed_payment_method': True, 'unknown_category': True, 'all_missing_values': True, 'minimum_roc_auc': True, 'minimum_f1': True}`
  - explanation: The implementation follows the specified requirements. The `build_model` function constructs a `sklearn.Pipeline` with a `ColumnTransformer` for preprocessing numeric and categorical features. Numeric features are imputed with the median and scaled. Categorical features are imputed with the most frequent value and encoded using `OneHotEncoder` with `handle_unknown='ignore'` to prevent errors on new categories. The final step is a `RandomForestClassifier` with the specified hyperparameters and `random_state` set for reproducibility. The `predict` function validates the input frame for required features and returns the probabilities for class 1, ensuring they are finite.

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
  "worker_seconds": 2.8551,
  "python_peak_mb": 53.574,
  "wall_seconds": 3.4455,
  "metrics": {
    "roc_auc": 0.802469135802469,
    "f1": 0.5507246376811594,
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
