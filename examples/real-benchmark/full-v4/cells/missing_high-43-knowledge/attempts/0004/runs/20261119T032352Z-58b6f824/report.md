# 算法能力验证报告

- Run: `20261119T032352Z-58b6f824`
- 状态：passed
- 模式：glm / glm-4.7-flash
- 故障注入：False
- 数据 SHA256：`e2b64050532ce82160943ac3e6cbb0504a4526df7c04b4cd4af2bc29ad0c796a`

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
        "algorithm": "logistic",
        "rationale": "逻辑回归作为可解释的线性基线模型，适合处理标准化的数值特征和类别不平衡问题。通过class_weight='balanced'处理少数类样本，确保模型在ROC-AUC和F1指标上的性能。",
        "hyperparameters": {
          "C": 1.0,
          "max_iter": 1000,
          "class_weight": "balanced",
          "solver": "lbfgs"
        }
      },
      {
        "algorithm": "forest",
        "rationale": "随机森林适合捕捉非线性关系和特征交互，通过限制树深度（max_depth=7）和叶节点样本数（min_samples_leaf=5）控制过拟合。class_weight='balanced'确保少数类权重，提升F1和ROC-AUC指标。",
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
| logistic | True | 2 | 0.7517942583732058 | 0.5794392523364486 |
| forest | True | 1 | 0.6749401913875599 | 0.5360824742268041 |

## 方案依据与修复轨迹

### logistic

逻辑回归作为可解释的线性基线模型，适合处理标准化的数值特征和类别不平衡问题。通过class_weight='balanced'处理少数类样本，确保模型在ROC-AUC和F1指标上的性能。

- attempt 0: passed=False, injected=False
  - code: `logistic/attempt_0/algorithm.py`; SHA256: `4e55fb9e1a4582ca061c28a823938a66c504cf36d197130a958d66ba4c6b0e53`
  - checks: `{}`
  - explanation: The implementation follows the specified interface and preprocessing plan. 
1. **Pipeline Structure**: A sklearn Pipeline is used with preprocessing steps and the classifier as the final step.
2. **Preprocessing**:
   - Numeric features ('tenure', 'monthly_charges', 'support_calls') are imputed with the median and scaled using `StandardScaler`.
   - Categorical features ('contract', 'payment_method') are imputed with the most frequent value and encoded using `OneHotEncoder` with `handle_unknown='ignore'` to handle unknown categories.
3. **Classifier**: `LogisticRegression` is used with `class_weight='balanced'` to handle class imbalance, as specified in the candidate hyperparameters.
4. **Interface**: `build_model(seed=42)` returns an unfitted Pipeline. `predict(model, frame)` validates required fields and returns the positive class probability.

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
  - code: `logistic/attempt_1/algorithm.py`; SHA256: `0ebf32ce58f32ddfb9ff04e7ff699765e40d7e3f13e8e0772c1844d652fd62d5`
  - checks: `{'probability_contract': True, 'estimator_matches_plan': True, 'hyperparameters_match_plan': True, 'preprocessing_matches_plan': True, 'missing_column_rejected': True, 'column_order': True, 'repeat_prediction': True, 'model_reload': True, 'unknown_category_contract': True, 'unknown_category_mixed_contract': True, 'unknown_category_payment_method': True, 'unknown_category_mixed_payment_method': True, 'unknown_category': True, 'all_missing_values': True, 'minimum_roc_auc': True, 'minimum_f1': True}`
  - explanation: The implementation uses a sklearn Pipeline with a ColumnTransformer to handle numeric and categorical features separately. Numeric features are imputed with the median and standardized. Categorical features are imputed with the most frequent value and encoded using OneHotEncoder with 'handle_unknown=ignore' to handle unknown categories. The final step is a LogisticRegression classifier with the specified hyperparameters, including 'class_weight=balanced' to address class imbalance. The 'predict' function checks for the presence of all required features and returns the probability of the positive class, ensuring finite values.
### forest

随机森林适合捕捉非线性关系和特征交互，通过限制树深度（max_depth=7）和叶节点样本数（min_samples_leaf=5）控制过拟合。class_weight='balanced'确保少数类权重，提升F1和ROC-AUC指标。

- attempt 0: passed=True, injected=False
  - code: `forest/attempt_0/algorithm.py`; SHA256: `6ce264782149f3830aa0ba891a468c6d7c06207f4c32102fb8e892711fab5c59`
  - checks: `{'probability_contract': True, 'estimator_matches_plan': True, 'hyperparameters_match_plan': True, 'preprocessing_matches_plan': True, 'missing_column_rejected': True, 'column_order': True, 'repeat_prediction': True, 'model_reload': True, 'unknown_category_contract': True, 'unknown_category_mixed_contract': True, 'unknown_category_payment_method': True, 'unknown_category_mixed_payment_method': True, 'unknown_category': True, 'all_missing_values': True, 'minimum_roc_auc': True, 'minimum_f1': True}`
  - explanation: The implementation follows the specified requirements:
1. Uses a sklearn Pipeline with the classifier as the final step
2. Implements the RandomForestClassifier with specified hyperparameters (n_estimators=100, max_depth=7, min_samples_leaf=5, class_weight='balanced')
3. Sets random_state=seed and n_jobs=1 as required
4. Handles missing values using median imputation for numeric features and most frequent for categorical
5. Standardizes numeric features using StandardScaler
6. Uses OneHotEncoder with handle_unknown='ignore' for categorical features
7. The build_model function returns an unfitted Pipeline
8. The predict function validates required fields and returns 1D positive-class probabilities
9. Implements all preprocessing within the pipeline to prevent data leakage

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
  "worker_seconds": 2.2592,
  "python_peak_mb": 50.887,
  "wall_seconds": 2.7417,
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
