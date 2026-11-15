# 算法能力验证报告

- Run: `20261115T030742Z-eb4a692d`
- 状态：passed
- 模式：mock / deterministic-templates-v1
- 故障注入：False
- 数据 SHA256：`4bc3314375da20388db5328d768f8c7c413c85c8393ee7aba181aeaca3223029`

## 需求

对客服工单文本进行紧急程度分类，label=1 表示紧急，0 表示普通。比较两个候选算法，处理缺失和空文本、未见过的词，使用 ROC-AUC、F1 和 balanced_accuracy，词表只在训练集拟合。

## 检索依据

知识抽取审计：status=passed，LLM 接受 0 条，拒绝 0 条；不合格事实未进入能力图谱。

工作流 v2；实验组=knowledge_repair；每候选生成预算=3；未向真实模型提供完整算法模板。

## 训练数据画像与实现计划

```json
{
  "data_profile": {
    "scope": "training_only",
    "rows": 480,
    "positive_rate": 0.5020833333333333,
    "minority_rate": 0.4979166666666667,
    "missing_rate": 0.027083333333333334,
    "mean_length": 19.38125,
    "requirements": [
      "train_only_preprocessing",
      "missing_values",
      "unseen_vocabulary"
    ]
  },
  "plan": {
    "candidates": [
      {
        "algorithm": "text_logistic",
        "rationale": "字符 TF-IDF 与线性模型，兼容中文工单。",
        "hyperparameters": {
          "C": 1.0,
          "max_iter": 1000,
          "class_weight": "balanced"
        }
      },
      {
        "algorithm": "text_nb",
        "rationale": "非负 TF-IDF 上的轻量分类基线，与线性模型比较。",
        "hyperparameters": {
          "alpha": 1.0,
          "fit_prior": true
        }
      }
    ],
    "preprocessing": {
      "ngram_max": 2,
      "max_features": 3000
    },
    "evidence_ids": [
      "Knowledge:261e56010fb50a81babe",
      "Knowledge:78a1b7dd32ededc8b25a",
      "Knowledge:41d89639e247933fa675",
      "Knowledge:4018437553590d695b11",
      "Knowledge:8c8cd3a7810e4b6a5290",
      "Knowledge:9bd6c46e9fb76d325348"
    ],
    "experience_ids": [],
    "experience_applied": false
  }
}
```

- 文本预处理与适用条件 — `text_business.md`:6（Knowledge:261e56010fb50a81babe）
  - 图谱路径：requirement:missing_values → strategy:imputation → Knowledge:261e56010fb50a81babe
  - 图谱路径：requirement:train_only_preprocessing → strategy:train_only_pipeline → Knowledge:261e56010fb50a81babe
- 缺失值和类别编码策略 — `churn_business.md`:8（Knowledge:78a1b7dd32ededc8b25a）
  - 图谱路径：requirement:missing_values → strategy:imputation → Knowledge:78a1b7dd32ededc8b25a
  - 图谱路径：requirement:train_only_preprocessing → strategy:train_only_pipeline → Knowledge:78a1b7dd32ededc8b25a
- 文本候选与评价 — `text_business.md`:9（Knowledge:41d89639e247933fa675）
- 候选算法及适用条件 — `churn_business.md`:11（Knowledge:4018437553590d695b11）
- 评价指标与业务约束 — `churn_business.md`:14（Knowledge:8c8cd3a7810e4b6a5290）
- 文本分类能力与接口 — `text_business.md`:3（Knowledge:9bd6c46e9fb76d325348）

历史经验检索数：0

## 候选比较（验证集）

| 算法 | 通过 | 尝试次数 | ROC-AUC | F1 |
|---|---|---|---|---|
| text_logistic | True | 1 | 0.8455227379278013 | 0.8220858895705522 |
| text_nb | True | 1 | 0.8369276449445227 | 0.8220858895705522 |

## 方案依据与修复轨迹

### text_logistic

字符 TF-IDF 与线性模型，兼容中文工单。

- attempt 0: passed=True, injected=False
  - code: `text_logistic/attempt_0/algorithm.py`; SHA256: `826cc920a872456bbd4dd8624b631cea5cb876f6f2f38e0d9498538b65fb375d`
  - checks: `{'probability_contract': True, 'estimator_matches_plan': True, 'hyperparameters_match_plan': True, 'preprocessing_matches_plan': True, 'missing_column_rejected': True, 'column_order': True, 'unseen_vocabulary': True, 'all_missing_values': True, 'repeat_prediction': True, 'model_reload': True, 'minimum_roc_auc': True, 'minimum_f1': True}`
  - explanation: Offline text template; not real LLM evidence
### text_nb

非负 TF-IDF 上的轻量分类基线，与线性模型比较。

- attempt 0: passed=True, injected=False
  - code: `text_nb/attempt_0/algorithm.py`; SHA256: `b195dbb5829d63d4404960cfa70fd23e2005eabfabe2553b4f5c5db48286e59f`
  - checks: `{'probability_contract': True, 'estimator_matches_plan': True, 'hyperparameters_match_plan': True, 'preprocessing_matches_plan': True, 'missing_column_rejected': True, 'column_order': True, 'unseen_vocabulary': True, 'all_missing_values': True, 'repeat_prediction': True, 'model_reload': True, 'minimum_roc_auc': True, 'minimum_f1': True}`
  - explanation: Offline text template; not real LLM evidence

## 最终测试集评估

选定算法：text_logistic

```json
{
  "checks": {
    "probability_contract": true,
    "estimator_matches_plan": true,
    "hyperparameters_match_plan": true,
    "preprocessing_matches_plan": true,
    "missing_column_rejected": true,
    "column_order": true,
    "unseen_vocabulary": true,
    "all_missing_values": true,
    "repeat_prediction": true,
    "model_reload": true,
    "minimum_roc_auc": true,
    "minimum_f1": true
  },
  "check_errors": {},
  "executed": true,
  "worker_seconds": 2.3861,
  "python_peak_mb": 51.441,
  "wall_seconds": 2.9545,
  "metrics": {
    "roc_auc": 0.8700781249999999,
    "f1": 0.8765432098765432,
    "accuracy": 0.875,
    "balanced_accuracy": 0.875
  },
  "passed": true
}
```

## 解释边界

数据为固定种子的模拟数据，指标不能代表真实客户业务效果。按验证集选择，最终测试结果不参与修复。
Mock 的需求解析、规划、代码生成和修复是规则/模板行为，不能作为真实大模型效果证据。
角色按状态机顺序交接；未实现自主并行多智能体，也未实现操作系统级安全沙箱。
python_peak_mb 仅统计 Python 内存分配，非进程总内存；wall_seconds 包含进程启动。
