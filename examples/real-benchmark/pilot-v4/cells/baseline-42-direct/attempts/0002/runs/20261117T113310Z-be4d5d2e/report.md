# 算法能力验证报告

- Run: `20261117T113310Z-be4d5d2e`
- 状态：failed
- 模式：glm / glm-4.7-flash
- 故障注入：False
- 数据 SHA256：`636769745426526c68263c70abce8af51e58eda7ea0ea72616830dc65fb7c50e`

## 需求

客户流失预测，处理缺失值、未知类别和类别不平衡，比较逻辑回归和随机森林，使用 ROC-AUC 和 F1 评价。

## 检索依据

运行错误：

```text
LLMProviderError: BigModel HTTP 429 [business_code=1305]: 模型访问量过大，请稍后重试
```

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
        "algorithm": "logistic",
        "rationale": "逻辑回归作为基线模型，适用于线性关系，且由于类别不平衡（36.1%），使用class_weight='balanced'进行权重调整。数值特征使用均值填充缺失值，分类特征使用众数填充。为了确保模型收敛，设置max_iter为1000，C参数在[0.01, 100]间搜索。",
        "hyperparameters": {
          "C": 1.0,
          "max_iter": 1000,
          "class_weight": "balanced",
          "solver": "lbfgs"
        }
      },
      {
        "algorithm": "forest",
        "rationale": "随机森林能够捕捉非线性关系和特征交互。针对小数据集（360行），设置max_depth=7和min_samples_leaf=5以防止过拟合。n_estimators设为100以平衡计算效率与性能。class_weight='balanced'处理类别不平衡。",
        "hyperparameters": {
          "n_estimators": 100,
          "max_depth": 7,
          "min_samples_leaf": 5,
          "class_weight": "balanced"
        }
      }
    ],
    "preprocessing": {
      "numeric_imputer": "mean",
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
| logistic | True | 1 | 0.785563273935367 | 0.6538461538461539 |
| forest | False | 2 | N/A | N/A |

## 方案依据与修复轨迹

### logistic

逻辑回归作为基线模型，适用于线性关系，且由于类别不平衡（36.1%），使用class_weight='balanced'进行权重调整。数值特征使用均值填充缺失值，分类特征使用众数填充。为了确保模型收敛，设置max_iter为1000，C参数在[0.01, 100]间搜索。

- attempt 0: passed=True, injected=False
  - code: `logistic/attempt_0/algorithm.py`; SHA256: `1f0122de08e5114c78a2cd08172202e99a31450e103ddabb86db581f9f35e532`
  - checks: `{'probability_contract': True, 'estimator_matches_plan': True, 'hyperparameters_match_plan': True, 'preprocessing_matches_plan': True, 'missing_column_rejected': True, 'column_order': True, 'repeat_prediction': True, 'model_reload': True, 'unknown_category_contract': True, 'unknown_category_mixed_contract': True, 'unknown_category_payment_method': True, 'unknown_category_mixed_payment_method': True, 'unknown_category': True, 'all_missing_values': True, 'minimum_roc_auc': True, 'minimum_f1': True}`
  - explanation: The implementation follows the specified requirements:

1. **Pipeline Structure**: Created a sklearn Pipeline with preprocessing steps followed by the classifier.

2. **Preprocessing**:
   - Numeric features: Mean imputation followed by StandardScaler
   - Categorical features: Most frequent imputation followed by OneHotEncoder (with 'ignore' for unknown categories)

3. **Classifier**: LogisticRegression with the specified hyperparameters:
   - random_state=42 for reproducibility
   - max_iter=1000 for convergence
   - class_weight='balanced' for handling class imbalance
   - solver='lbfgs' and C=1.0 as specified

4. **Prediction**:
   - Checks for all required features before prediction
   - Returns the positive class probabilities (column 1)
   - Validates that predictions are finite values
   - Raises ValueError for missing features or invalid predictions

The model is designed for the churn classification task with the specified handling of missing values, unknown categories, and class imbalance.
### forest

随机森林能够捕捉非线性关系和特征交互。针对小数据集（360行），设置max_depth=7和min_samples_leaf=5以防止过拟合。n_estimators设为100以平衡计算效率与性能。class_weight='balanced'处理类别不平衡。

- attempt 0: passed=False, injected=False
  - code: `forest/attempt_0/algorithm.py`; SHA256: `22dd70bb065473dac5c5051c884a15be33e3d576f898c7591efe70771c92457e`
  - checks: `{}`
  - explanation: The implementation creates a sklearn Pipeline with a preprocessor and RandomForestClassifier. The preprocessor handles missing values using mean for numeric and most frequent for categorical features, and applies one-hot encoding to categorical variables with unknown categories handled via 'ignore'. The RandomForestClassifier is configured with the specified hyperparameters (n_estimators=100, max_depth=7, min_samples_leaf=5, class_weight='balanced') and a random_state for reproducibility. The predict function validates input features and returns positive class probabilities.

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
  - code: `forest/attempt_1/algorithm.py`; SHA256: `9f9e8bf8b2158dc714c0ac882c5ffa307f58d3339310c29c7c336d1e23418b8a`
  - checks: `{}`
  - explanation: The implementation creates a scikit-learn Pipeline with a preprocessing step (handling missing values for numeric features with mean imputation and categorical features with most frequent imputation, followed by One-Hot Encoding) and a RandomForestClassifier as the final step. The classifier is configured with the specified hyperparameters (100 estimators, max_depth=7, min_samples_leaf=5, class_weight='balanced') and a fixed random_state for reproducibility. The predict function validates the input frame for required features, raises a ValueError if any are missing, and returns the probability of the positive class (class 1) in the same order as the input rows. It also checks for non-finite values in the predictions.

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


## 最终测试集评估

选定算法：None

```json
null
```

## 解释边界

数据为固定种子的模拟数据，指标不能代表真实客户业务效果。按验证集选择，最终测试结果不参与修复。
Mock 的需求解析、规划、代码生成和修复是规则/模板行为，不能作为真实大模型效果证据。
角色按状态机顺序交接；未实现自主并行多智能体，也未实现操作系统级安全沙箱。
python_peak_mb 仅统计 Python 内存分配，非进程总内存；wall_seconds 包含进程启动。
