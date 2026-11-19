# 算法能力验证报告

- Run: `20261117T163829Z-d20eb4ed`
- 状态：failed
- 模式：glm / glm-4.7-flash
- 故障注入：False
- 数据 SHA256：`23c681619201b59f721b4b7f1b6662c908bd945db5798381c0ef968df86a1681`

## 需求

客户流失预测，处理缺失值、未知类别和类别不平衡，比较逻辑回归和随机森林，使用 ROC-AUC 和 F1 评价。

## 检索依据

运行错误：

```text
ValueError: plan contract rejected after 2 responses: Plan cites unknown evidence_ids
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
          "0.5": 36.5,
          "0.75": 55.25
        }
      },
      "monthly_charges": {
        "kind": "numeric",
        "missing_rate": 0.016667,
        "cardinality": 354,
        "constant": false,
        "quantiles": {
          "0.25": 41.53662006741646,
          "0.5": 68.45100204378839,
          "0.75": 95.30866672610692
        }
      },
      "support_calls": {
        "kind": "numeric",
        "missing_rate": 0.027778,
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
        "missing_rate": 0.016667,
        "cardinality": 3,
        "constant": false
      },
      "payment_method": {
        "kind": "categorical",
        "missing_rate": 0.025,
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
