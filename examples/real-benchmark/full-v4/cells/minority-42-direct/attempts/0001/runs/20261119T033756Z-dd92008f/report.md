# 算法能力验证报告

- Run: `20261119T033756Z-dd92008f`
- 状态：passed
- 模式：glm / glm-4.7-flash
- 故障注入：False
- 数据 SHA256：`5bac8d4d229c12bfae963c6bb6603450050beee8492e10acac43708b99b05390`

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
        "rationale": "基于线性模型的可解释性作为基线。数据包含 3 个数值特征（tenure, monthly_charges, support_calls）和 2 个分类特征（contract, payment_method）。数值特征缺失率约 7.5%-9%，适合使用均值（mean）或中位数（median）填充。由于任务要求比较模型且数据集较小（360行），逻辑回归是标准基线。使用类别权重 'balanced' 处理 22.2% 的少数类样本。数值特征需标准化以使 L2 正则化（默认 C=1.0）有效。",
        "hyperparameters": {
          "C": 1.0,
          "max_iter": 1000,
          "class_weight": "balanced",
          "solver": "lbfgs"
        }
      },
      {
        "algorithm": "forest",
        "rationale": "随机森林能够捕捉特征间的非线性交互和复杂模式，适合比较。数值特征缺失率约 7.5%-9%，使用中位数（median）填充。分类特征缺失率约 10%，使用众数（most_frequent）填充。由于数据集较小，设置较低的树深度（max_depth）和较高的最小叶子样本数（min_samples_leaf）以防止过拟合。类别权重 'balanced' 处理少数类不平衡。",
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
| logistic | True | 2 | 0.7610513739545997 | 0.4810126582278481 |
| forest | True | 1 | 0.712863401035444 | 0.47368421052631576 |

## 方案依据与修复轨迹

### logistic

基于线性模型的可解释性作为基线。数据包含 3 个数值特征（tenure, monthly_charges, support_calls）和 2 个分类特征（contract, payment_method）。数值特征缺失率约 7.5%-9%，适合使用均值（mean）或中位数（median）填充。由于任务要求比较模型且数据集较小（360行），逻辑回归是标准基线。使用类别权重 'balanced' 处理 22.2% 的少数类样本。数值特征需标准化以使 L2 正则化（默认 C=1.0）有效。

- attempt 0: passed=False, injected=False
  - code: `logistic/attempt_0/algorithm.py`; SHA256: `1060e99b0a965dba1d9980a411e0222f1e993eb3666cad00862ca02a970db507`
  - checks: `{}`
  - explanation: The implementation creates a sklearn Pipeline with a preprocessing step for numeric and categorical features and a LogisticRegression classifier as the final step. Numeric features are imputed with the median, scaled with StandardScaler, and passed through. Categorical features are imputed with the most frequent value, one-hot encoded, and passed through. The classifier is configured with the provided hyperparameters: C=1.0, max_iter=1000, class_weight='balanced', and solver='lbfgs'. The build_model function returns this unfitted pipeline. The predict function checks for required fields, raises a ValueError if any are missing, and returns the positive class probabilities. It also checks for finite values and raises a ValueError if any are not finite.

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
  - code: `logistic/attempt_1/algorithm.py`; SHA256: `84572baffc4010f545284709e98321afe541fe4100d1b6a59d123d48448ae13f`
  - checks: `{'probability_contract': True, 'estimator_matches_plan': True, 'hyperparameters_match_plan': True, 'preprocessing_matches_plan': True, 'missing_column_rejected': True, 'column_order': True, 'repeat_prediction': True, 'model_reload': True, 'unknown_category_contract': True, 'unknown_category_mixed_contract': True, 'unknown_category_payment_method': True, 'unknown_category_mixed_payment_method': True, 'unknown_category': True, 'all_missing_values': True, 'minimum_roc_auc': True, 'minimum_f1': True}`
  - explanation: The implementation creates a scikit-learn Pipeline for the Logistic Regression model. It uses ColumnTransformer to apply median imputation and scaling to numeric features, and most frequent imputation and OneHotEncoding to categorical features. The classifier is configured with the specified hyperparameters, including class_weight='balanced' to handle the class imbalance and random_state for reproducibility. The predict function validates input for required fields and returns the positive class probabilities.
### forest

随机森林能够捕捉特征间的非线性交互和复杂模式，适合比较。数值特征缺失率约 7.5%-9%，使用中位数（median）填充。分类特征缺失率约 10%，使用众数（most_frequent）填充。由于数据集较小，设置较低的树深度（max_depth）和较高的最小叶子样本数（min_samples_leaf）以防止过拟合。类别权重 'balanced' 处理少数类不平衡。

- attempt 0: passed=True, injected=False
  - code: `forest/attempt_0/algorithm.py`; SHA256: `6b2df01ceb36599a54f544b5361b5694debffcf27756f989447a7ea193acddda`
  - checks: `{'probability_contract': True, 'estimator_matches_plan': True, 'hyperparameters_match_plan': True, 'preprocessing_matches_plan': True, 'missing_column_rejected': True, 'column_order': True, 'repeat_prediction': True, 'model_reload': True, 'unknown_category_contract': True, 'unknown_category_mixed_contract': True, 'unknown_category_payment_method': True, 'unknown_category_mixed_payment_method': True, 'unknown_category': True, 'all_missing_values': True, 'minimum_roc_auc': True, 'minimum_f1': True}`
  - explanation: The implementation creates a scikit-learn Pipeline. The first step handles preprocessing: numeric features are imputed with the median and scaled using StandardScaler. Categorical features are imputed with the most frequent value and one-hot encoded with 'ignore' for unknown categories. The final step is the RandomForestClassifier with the specified hyperparameters (n_estimators=100, max_depth=7, min_samples_leaf=5, class_weight='balanced') to address class imbalance. The `build_model` function returns the unfitted pipeline. The `predict` function checks for required features and returns the probability of the positive class (class 1).

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
  "worker_seconds": 2.2145,
  "python_peak_mb": 51.109,
  "wall_seconds": 2.7223,
  "metrics": {
    "roc_auc": 0.7956989247311828,
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
