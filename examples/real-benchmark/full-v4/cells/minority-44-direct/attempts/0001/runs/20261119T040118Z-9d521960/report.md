# 算法能力验证报告

- Run: `20261119T040118Z-9d521960`
- 状态：failed
- 模式：glm / glm-4.7-flash
- 故障注入：False
- 数据 SHA256：`a80959c53c0b16bd2c0347537dcd64f175d093ddc91106c6ba51dd4bf36a0100`

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
        "missing_rate": 0.063889,
        "cardinality": 71,
        "constant": false,
        "quantiles": {
          "0.25": 19.0,
          "0.5": 37.0,
          "0.75": 55.0
        }
      },
      "monthly_charges": {
        "kind": "numeric",
        "missing_rate": 0.047222,
        "cardinality": 343,
        "constant": false,
        "quantiles": {
          "0.25": 42.54325333506832,
          "0.5": 69.47080129149407,
          "0.75": 94.84542815373783
        }
      },
      "support_calls": {
        "kind": "numeric",
        "missing_rate": 0.083333,
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
        "missing_rate": 0.066667,
        "cardinality": 3,
        "constant": false
      },
      "payment_method": {
        "kind": "categorical",
        "missing_rate": 0.1,
        "cardinality": 2,
        "constant": false
      }
    },
    "positive_rate": 0.20277777777777778,
    "minority_rate": 0.20277777777777778,
    "requirements": [
      "train_only_preprocessing",
      "unknown_categories",
      "missing_values",
      "class_imbalance"
    ]
  },
  "plan": null
}
```


历史经验检索数：0

## 候选比较（验证集）

| 算法 | 通过 | 尝试次数 | ROC-AUC | F1 |
|---|---|---|---|---|

## 方案依据与修复轨迹


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
