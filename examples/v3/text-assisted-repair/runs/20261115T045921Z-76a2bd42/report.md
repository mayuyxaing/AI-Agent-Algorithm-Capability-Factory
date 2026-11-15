# 算法能力验证报告

- Run: `20261115T045921Z-76a2bd42`
- 状态：passed
- 模式：assisted_glm / glm-4.7-flash
- 故障注入：False
- 数据 SHA256：`4bc3314375da20388db5328d768f8c7c413c85c8393ee7aba181aeaca3223029`

## 需求

对客服工单文本进行紧急程度分类，label=1 表示紧急，0 表示普通。比较两个候选算法，处理缺失和空文本、未见过的词，使用 ROC-AUC、F1 和 balanced_accuracy，词表只在训练集拟合。

## 检索依据

## 响应来源与辅助修改

```json
{
  "kind": "assisted_reproduction",
  "source_run_id": "20261115T044624Z-10c203ec",
  "original_status": "no_valid_candidate",
  "new_api_calls": 0,
  "modification_author": "coding assistant, outside the GLM workflow",
  "change": "Select the text column when the cleaner receives a DataFrame; no estimator/parameter/metric changes",
  "changes": [
    {
      "algorithm": "text_logistic",
      "source_code_path": "text_logistic/attempt_2/algorithm.py",
      "source_sha256": "0f2f9d5751e19ab319c27edd4180bfd17bbe09aa8fec44fb999d857a8bd5326e",
      "patched_sha256": "609180ec862b51416c9dfd66b97e882d7d10fcf7428574977b9505f286288ef1"
    },
    {
      "algorithm": "text_nb",
      "source_code_path": "text_nb/attempt_2/algorithm.py",
      "source_sha256": "c7d46b96df0906658ae9deee004bc7c6e2e62a09eea52418e0ac11460744f08c",
      "patched_sha256": "3787299af078e1ea59b82b76c6949f3a5663bc1e4677b7e9c452a9efc045df06"
    }
  ]
}
```

知识抽取审计：status=passed，LLM 接受 20 条，拒绝 0 条；不合格事实未进入能力图谱。

工作流 v2；实验组=knowledge_repair；每候选生成预算=1；未向真实模型提供完整算法模板。

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
        "rationale": "逻辑回归是线性模型，与字符级TF-IDF特征结合效果良好，且能处理中文文本。相比朴素贝叶斯，它对未见词汇的处理能力更强，且不需要严格的非负特征假设。使用class_weight='balanced'处理正负样本不平衡。",
        "hyperparameters": {
          "C": 1.0,
          "max_iter": 1000,
          "class_weight": "balanced"
        }
      },
      {
        "algorithm": "text_nb",
        "rationale": "多项式朴素贝叶斯作为基线模型，适合处理词袋特征，计算效率高。但它要求特征非负，且对未见词汇的概率估计可能不如逻辑回归稳健。",
        "hyperparameters": {
          "alpha": 1.0,
          "fit_prior": true
        }
      }
    ],
    "preprocessing": {
      "ngram_max": 2,
      "max_features": 2000
    },
    "evidence_ids": [
      "Knowledge:261e56010fb50a81babe",
      "Knowledge:e11b360accf27c182e8b",
      "Knowledge:0fedab0953762cb5b063",
      "Knowledge:24ac209bc38141c65561"
    ],
    "experience_ids": [],
    "experience_applied": false
  }
}
```

- 文本预处理与适用条件 — `text_business.md`:6（Knowledge:261e56010fb50a81babe）
  - 图谱路径：requirement:missing_values → strategy:imputation → Knowledge:261e56010fb50a81babe
  - 图谱路径：requirement:train_only_preprocessing → strategy:train_only_pipeline → Knowledge:261e56010fb50a81babe
  - 图谱路径：requirement:unseen_vocabulary → strategy:train_fitted_tfidf → Knowledge:261e56010fb50a81babe
- 预处理方法 — `text_business.md`:6（Knowledge:e11b360accf27c182e8b）
  - 图谱路径：requirement:missing_values → strategy:imputation → Knowledge:e11b360accf27c182e8b
  - 图谱路径：requirement:train_only_preprocessing → strategy:train_only_pipeline → Knowledge:e11b360accf27c182e8b
  - 图谱路径：requirement:unseen_vocabulary → strategy:train_fitted_tfidf → Knowledge:e11b360accf27c182e8b
- 适用条件 — `text_business.md`:6（Knowledge:0fedab0953762cb5b063）
  - 图谱路径：requirement:missing_values → strategy:imputation → Knowledge:0fedab0953762cb5b063
  - 图谱路径：requirement:train_only_preprocessing → strategy:train_only_pipeline → Knowledge:0fedab0953762cb5b063
  - 图谱路径：requirement:unseen_vocabulary → strategy:train_fitted_tfidf → Knowledge:0fedab0953762cb5b063
- 特征处理 — `text_business.md`:9（Knowledge:24ac209bc38141c65561）
  - 图谱路径：requirement:train_only_preprocessing → strategy:train_only_pipeline → Knowledge:24ac209bc38141c65561
  - 图谱路径：requirement:unseen_vocabulary → strategy:train_fitted_tfidf → Knowledge:24ac209bc38141c65561
- 缺失值和类别编码策略 — `churn_business.md`:8（Knowledge:78a1b7dd32ededc8b25a）
  - 图谱路径：requirement:missing_values → strategy:imputation → Knowledge:78a1b7dd32ededc8b25a
  - 图谱路径：requirement:train_only_preprocessing → strategy:train_only_pipeline → Knowledge:78a1b7dd32ededc8b25a
- 特征缺失值处理 — `churn_business.md`:8（Knowledge:69e822e936d0938355b4）
  - 图谱路径：requirement:missing_values → strategy:imputation → Knowledge:69e822e936d0938355b4
  - 图谱路径：requirement:train_only_preprocessing → strategy:train_only_pipeline → Knowledge:69e822e936d0938355b4

历史经验检索数：1

## 候选比较（验证集）

| 算法 | 通过 | 尝试次数 | ROC-AUC | F1 |
|---|---|---|---|---|
| text_logistic | True | 1 | 0.8455227379278013 | 0.8220858895705522 |
| text_nb | True | 1 | 0.8369276449445227 | 0.8220858895705522 |

## 方案依据与修复轨迹

### text_logistic

逻辑回归是线性模型，与字符级TF-IDF特征结合效果良好，且能处理中文文本。相比朴素贝叶斯，它对未见词汇的处理能力更强，且不需要严格的非负特征假设。使用class_weight='balanced'处理正负样本不平衡。

- attempt 0: passed=True, injected=False
  - code: `text_logistic/attempt_0/algorithm.py`; SHA256: `609180ec862b51416c9dfd66b97e882d7d10fcf7428574977b9505f286288ef1`
  - checks: `{'probability_contract': True, 'estimator_matches_plan': True, 'hyperparameters_match_plan': True, 'preprocessing_matches_plan': True, 'missing_column_rejected': True, 'column_order': True, 'repeat_prediction': True, 'model_reload': True, 'unseen_vocabulary': True, 'all_missing_values': True, 'minimum_roc_auc': True, 'minimum_f1': True}`
  - explanation: Recorded GLM source with an explicit coding-assistant DataFrame boundary patch; no new API call or autonomous GLM repair.
### text_nb

多项式朴素贝叶斯作为基线模型，适合处理词袋特征，计算效率高。但它要求特征非负，且对未见词汇的概率估计可能不如逻辑回归稳健。

- attempt 0: passed=True, injected=False
  - code: `text_nb/attempt_0/algorithm.py`; SHA256: `3787299af078e1ea59b82b76c6949f3a5663bc1e4677b7e9c452a9efc045df06`
  - checks: `{'probability_contract': True, 'estimator_matches_plan': True, 'hyperparameters_match_plan': True, 'preprocessing_matches_plan': True, 'missing_column_rejected': True, 'column_order': True, 'repeat_prediction': True, 'model_reload': True, 'unseen_vocabulary': True, 'all_missing_values': True, 'minimum_roc_auc': True, 'minimum_f1': True}`
  - explanation: Recorded GLM source with an explicit coding-assistant DataFrame boundary patch; no new API call or autonomous GLM repair.

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
    "repeat_prediction": true,
    "model_reload": true,
    "unseen_vocabulary": true,
    "all_missing_values": true,
    "minimum_roc_auc": true,
    "minimum_f1": true
  },
  "check_errors": {},
  "executed": true,
  "worker_seconds": 2.2886,
  "python_peak_mb": 51.365,
  "wall_seconds": 2.7783,
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
