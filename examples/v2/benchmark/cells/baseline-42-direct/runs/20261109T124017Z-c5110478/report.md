# 算法能力验证报告

- Run: `20261109T124017Z-c5110478`
- 状态：passed
- 模式：mock / deterministic-templates-v1
- 故障注入：False
- 数据 SHA256：`636769745426526c68263c70abce8af51e58eda7ea0ea72616830dc65fb7c50e`

## 需求

客户流失预测，处理缺失值、未知类别和类别不平衡，比较逻辑回归和随机森林，使用 ROC-AUC 和 F1 评价。

## 检索依据

知识抽取审计：status=passed，LLM 接受 0 条，拒绝 0 条；不合格事实未进入能力图谱。

复用了 0 次历史模型响应，来源运行见 report.json；算法验证结果没有缓存。

工作流 v2；实验组=direct；每候选生成预算=3；未向真实模型提供完整算法模板。

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
        "algorithm": "forest",
        "rationale": "与线性模型比较非线性关系和特征交互；限制树深度与并行度。 训练样本数=360，少数类比例=0.361。",
        "hyperparameters": {
          "n_estimators": 100,
          "max_depth": 5,
          "min_samples_leaf": 5,
          "class_weight": "balanced"
        }
      },
      {
        "algorithm": "logistic",
        "rationale": "可解释线性基线；标准化数值特征，使用类别权重处理不平衡。 训练样本数=360，少数类比例=0.361。",
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
| forest | True | 1 | 0.7547568710359408 | 0.6238532110091743 |
| logistic | True | 1 | 0.7864693446088795 | 0.6470588235294118 |

## 方案依据与修复轨迹

### forest

与线性模型比较非线性关系和特征交互；限制树深度与并行度。 训练样本数=360，少数类比例=0.361。

- attempt 0: passed=True, injected=False
  - code: `forest/attempt_0/algorithm.py`; SHA256: `43392bf512d2f9ec48bf063d9e474811e5c6e3ab72fc2ced5b7902f72c6a9a03`
  - checks: `{'probability_contract': True, 'estimator_matches_plan': True, 'hyperparameters_match_plan': True, 'preprocessing_matches_plan': True, 'missing_column_rejected': True, 'column_order': True, 'unknown_category': True, 'all_missing_values': True, 'repeat_prediction': True, 'model_reload': True, 'minimum_roc_auc': True, 'minimum_f1': True}`
  - explanation: Offline template generation
### logistic

可解释线性基线；标准化数值特征，使用类别权重处理不平衡。 训练样本数=360，少数类比例=0.361。

- attempt 0: passed=True, injected=False
  - code: `logistic/attempt_0/algorithm.py`; SHA256: `594629e008064c295774704811156427ab3b6df8d2f76969a2410e906f217ea2`
  - checks: `{'probability_contract': True, 'estimator_matches_plan': True, 'hyperparameters_match_plan': True, 'preprocessing_matches_plan': True, 'missing_column_rejected': True, 'column_order': True, 'unknown_category': True, 'all_missing_values': True, 'repeat_prediction': True, 'model_reload': True, 'minimum_roc_auc': True, 'minimum_f1': True}`
  - explanation: Offline template generation

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
    "unknown_category": true,
    "all_missing_values": true,
    "repeat_prediction": true,
    "model_reload": true,
    "minimum_roc_auc": true,
    "minimum_f1": true
  },
  "check_errors": {},
  "executed": true,
  "worker_seconds": 2.1286,
  "python_peak_mb": 50.789,
  "wall_seconds": 2.9065,
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
