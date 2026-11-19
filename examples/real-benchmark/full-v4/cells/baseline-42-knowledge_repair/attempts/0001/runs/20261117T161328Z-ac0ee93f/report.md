# 算法能力验证报告

- Run: `20261117T161328Z-ac0ee93f`
- 状态：passed
- 模式：glm / glm-4.7-flash
- 故障注入：False
- 数据 SHA256：`636769745426526c68263c70abce8af51e58eda7ea0ea72616830dc65fb7c50e`

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
        "rationale": "逻辑回归作为线性基线模型，适合处理具有缺失值和类别不平衡的客户流失预测任务。使用训练集的中位数填补缺失数值特征，众数填补缺失类别特征，并进行标准化以适应模型。由于类别不平衡（少数类占比约36%），设置 class_weight='balanced' 来调整权重，确保模型对少数类有足够的重视。使用 LBFGS 求解器，并限制最大迭代次数为 1000，以防止过拟合。此模型可解释性强，能够提供特征权重的直观解释。",
        "hyperparameters": {
          "C": 1.0,
          "max_iter": 1000,
          "class_weight": "balanced",
          "solver": "lbfgs"
        }
      },
      {
        "algorithm": "forest",
        "rationale": "随机森林能够捕捉非线性关系和特征交互，适合处理复杂的客户流失数据。通过限制树深度（max_depth=7）和最小叶子节点样本数（min_samples_leaf=5）来控制过拟合。设置 class_weight='balanced' 处理类别不平衡问题。使用 100 棵树（n_estimators=100）并设置 n_jobs=-1 以并行计算，提高训练效率。此模型在 ROC-AUC 和 F1 指标上通常表现优于逻辑回归，尤其是在特征交互复杂的情况下。",
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
| logistic | True | 1 | 0.7864693446088795 | 0.6470588235294118 |
| forest | True | 1 | 0.7363334340078526 | 0.6095238095238096 |

## 方案依据与修复轨迹

### logistic

逻辑回归作为线性基线模型，适合处理具有缺失值和类别不平衡的客户流失预测任务。使用训练集的中位数填补缺失数值特征，众数填补缺失类别特征，并进行标准化以适应模型。由于类别不平衡（少数类占比约36%），设置 class_weight='balanced' 来调整权重，确保模型对少数类有足够的重视。使用 LBFGS 求解器，并限制最大迭代次数为 1000，以防止过拟合。此模型可解释性强，能够提供特征权重的直观解释。

- attempt 0: passed=True, injected=False
  - code: `logistic/attempt_0/algorithm.py`; SHA256: `8f5c336b51a04a055c85dad70b6715cc169fc960ab161eb74ff159f53a410d5d`
  - checks: `{'probability_contract': True, 'estimator_matches_plan': True, 'hyperparameters_match_plan': True, 'preprocessing_matches_plan': True, 'missing_column_rejected': True, 'column_order': True, 'repeat_prediction': True, 'model_reload': True, 'unknown_category_contract': True, 'unknown_category_mixed_contract': True, 'unknown_category_payment_method': True, 'unknown_category_mixed_payment_method': True, 'unknown_category': True, 'all_missing_values': True, 'minimum_roc_auc': True, 'minimum_f1': True}`
  - explanation: The implementation builds a sklearn Pipeline with the following steps:
1. **Preprocessing (ColumnTransformer)**:
   - **Numeric features** ('tenure', 'monthly_charges', 'support_calls') are processed by a pipeline: `SimpleImputer(strategy='median')` handles missing values, followed by `StandardScaler()` for normalization.
   - **Categorical features** ('contract', 'payment_method') are processed by a pipeline: `SimpleImputer(strategy='most_frequent')` fills missing values with the most frequent category, and `OneHotEncoder(handle_unknown='ignore')` encodes categories while ignoring any new categories during prediction.
2. **Classifier**: `LogisticRegression` is used as the final step with the specified hyperparameters (`C=1.0`, `max_iter=1000`, `class_weight='balanced'`, `solver='lbfgs'`, `random_state=42`) to handle class imbalance.
3. **Prediction**: The `predict` function checks for required features, raises a `ValueError` if any are missing, and returns the probability of the positive class (index 1) using `model.predict_proba`.
### forest

随机森林能够捕捉非线性关系和特征交互，适合处理复杂的客户流失数据。通过限制树深度（max_depth=7）和最小叶子节点样本数（min_samples_leaf=5）来控制过拟合。设置 class_weight='balanced' 处理类别不平衡问题。使用 100 棵树（n_estimators=100）并设置 n_jobs=-1 以并行计算，提高训练效率。此模型在 ROC-AUC 和 F1 指标上通常表现优于逻辑回归，尤其是在特征交互复杂的情况下。

- attempt 0: passed=True, injected=False
  - code: `forest/attempt_0/algorithm.py`; SHA256: `50eff92c6c98536551061945e6e564d4bbd8af822eeb49ff8b623aedb285764f`
  - checks: `{'probability_contract': True, 'estimator_matches_plan': True, 'hyperparameters_match_plan': True, 'preprocessing_matches_plan': True, 'missing_column_rejected': True, 'column_order': True, 'repeat_prediction': True, 'model_reload': True, 'unknown_category_contract': True, 'unknown_category_mixed_contract': True, 'unknown_category_payment_method': True, 'unknown_category_mixed_payment_method': True, 'unknown_category': True, 'all_missing_values': True, 'minimum_roc_auc': True, 'minimum_f1': True}`
  - explanation: The implementation follows the specified interface and requirements. It uses a sklearn Pipeline with a ColumnTransformer for preprocessing, handling missing values (median for numeric, most frequent for categorical), standardizing numeric features, and one-hot encoding categorical features with unknown category handling. The final classifier is a RandomForestClassifier with the specified hyperparameters and random_state for reproducibility. The predict function validates input features and returns the positive class probabilities.

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
  "worker_seconds": 2.1988,
  "python_peak_mb": 50.904,
  "wall_seconds": 2.6347,
  "metrics": {
    "roc_auc": 0.879492600422833,
    "f1": 0.72,
    "accuracy": 0.7666666666666667
  },
  "passed": true
}
```

## 解释边界

数据为固定种子的模拟数据，指标不能代表真实客户业务效果。按验证集选择，最终测试结果不参与修复。
Mock 的需求解析、规划、代码生成和修复是规则/模板行为，不能作为真实大模型效果证据。
角色按状态机顺序交接；未实现自主并行多智能体，也未实现操作系统级安全沙箱。
python_peak_mb 仅统计 Python 内存分配，非进程总内存；wall_seconds 包含进程启动。
