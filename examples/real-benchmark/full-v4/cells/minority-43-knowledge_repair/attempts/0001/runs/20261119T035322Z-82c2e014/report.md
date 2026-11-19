# 算法能力验证报告

- Run: `20261119T035322Z-82c2e014`
- 状态：failed
- 模式：glm / glm-4.7-flash
- 故障注入：False
- 数据 SHA256：`10ff7f6cf2641f927c954282bea4208e1611532510669eb5056503e1f3d0bafe`

## 需求

客户流失预测，处理缺失值、未知类别和类别不平衡，比较逻辑回归和随机森林，使用 ROC-AUC 和 F1 评价。

## 检索依据

运行错误：

```text
LLMProviderError: BigModel HTTP 500: 服务异常，请稍后重试
```

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
        "rationale": "逻辑回归作为可解释的线性基线模型，配合 class_weight='balanced' 处理类别不平衡。根据数据特征（数值特征缺失率约10%，类别特征缺失率约7%），使用中位数填补数值特征并标准化，众数填补类别特征，并使用独热编码允许未知类别。",
        "hyperparameters": {
          "C": 1.0,
          "max_iter": 1000,
          "class_weight": "balanced",
          "solver": "lbfgs"
        }
      },
      {
        "algorithm": "forest",
        "rationale": "随机森林适合捕捉非线性关系和特征交互。在处理类别不平衡时使用 class_weight='balanced'。限制树深度（max_depth=7）和最小叶子样本数（min_samples_leaf=5）以控制过拟合，并设置 n_estimators=100 以平衡性能和计算资源。",
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
      "Knowledge:948b7c74dfbb766ab40d",
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
| logistic | True | 2 | 0.7903157894736843 | 0.5396825396825397 |
| forest | False | 0 | N/A | N/A |

## 方案依据与修复轨迹

### logistic

逻辑回归作为可解释的线性基线模型，配合 class_weight='balanced' 处理类别不平衡。根据数据特征（数值特征缺失率约10%，类别特征缺失率约7%），使用中位数填补数值特征并标准化，众数填补类别特征，并使用独热编码允许未知类别。

- attempt 0: passed=False, injected=False
  - code: `logistic/attempt_0/algorithm.py`; SHA256: `5f6f9fc25a78a3252412e114e32c57232f4e5fc8c65db7fff952a8eae6eaf764`
  - checks: `{'static_check': False}`
  - explanation: The implementation follows the specified contract and requirements. The `build_model` function creates a scikit-learn Pipeline that includes a custom transformer to validate the presence of required features, preprocessing steps for both numeric and categorical features (including handling missing values and unknown categories), and a LogisticRegression classifier with the provided hyperparameters. The `predict` function takes a fitted pipeline and a data frame, validates the required fields, and returns the probability of the positive class. The code adheres to the constraints of using only the allowed imports, avoiding module-level training, and ensuring the model is returned in an unfitted state.

```text
Prohibited name: __name__
```

- attempt 1: passed=True, injected=False
  - code: `logistic/attempt_1/algorithm.py`; SHA256: `e3accfbfac434dde07acfba42ef401a0b2f68b5e772f232a1beac69460d5d3ea`
  - checks: `{'probability_contract': True, 'estimator_matches_plan': True, 'hyperparameters_match_plan': True, 'preprocessing_matches_plan': True, 'missing_column_rejected': True, 'column_order': True, 'repeat_prediction': True, 'model_reload': True, 'unknown_category_contract': True, 'unknown_category_mixed_contract': True, 'unknown_category_payment_method': True, 'unknown_category_mixed_payment_method': True, 'unknown_category': True, 'all_missing_values': True, 'minimum_roc_auc': True, 'minimum_f1': True}`
  - explanation: The provided code implements a scikit-learn pipeline for churn classification that strictly adheres to the specified requirements. It uses a custom transformer to enforce the presence of required features, a ColumnTransformer to apply separate preprocessing pipelines to numeric and categorical features (median imputation and scaling for numeric, most frequent imputation and one-hot encoding with unknown category handling for categorical), and a LogisticRegression classifier with balanced class weights and a fixed random state. The `predict` function validates input data and returns the probability of the positive class. The code is fully contained within the module, avoiding module-level execution or dynamic imports.
### forest

随机森林适合捕捉非线性关系和特征交互。在处理类别不平衡时使用 class_weight='balanced'。限制树深度（max_depth=7）和最小叶子样本数（min_samples_leaf=5）以控制过拟合，并设置 n_estimators=100 以平衡性能和计算资源。


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
