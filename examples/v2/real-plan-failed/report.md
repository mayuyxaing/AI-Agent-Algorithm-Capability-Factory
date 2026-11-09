# 算法能力验证报告

- Run: `20261109T125442Z-e8ddcbed`
- 状态：failed
- 模式：glm / glm-4.7-flash
- 故障注入：True
- 数据 SHA256：`876184b02cc537285eabc94ddfccda984f0ed35ba043b52e97b502a56b6be483`

## 需求

根据客户使用时长、月费用、客服次数、合同类型和支付方式预测客户流失。处理缺失值与类别不平衡，比较逻辑回归和随机森林，使用 ROC-AUC 和 F1 评价，提供统一预测接口，能够处理未知类别。

## 检索依据

运行错误：

```text
ValueError: Plan must contain 1 to 3 candidates
```

知识抽取审计：status=passed，LLM 接受 12 条，拒绝 0 条；不合格事实未进入能力图谱。

复用了 1 次历史模型响应，来源运行见 report.json；算法验证结果没有缓存。

工作流 v2；实验组=knowledge_repair；每候选生成预算=3；未向真实模型提供完整算法模板。

## 训练数据画像与实现计划

```json
{
  "data_profile": {
    "scope": "training_only",
    "rows": 720,
    "columns": {
      "tenure": {
        "kind": "numeric",
        "missing_rate": 0.038889,
        "cardinality": 72,
        "constant": false,
        "quantiles": {
          "0.25": 20.0,
          "0.5": 37.0,
          "0.75": 56.0
        }
      },
      "monthly_charges": {
        "kind": "numeric",
        "missing_rate": 0.036111,
        "cardinality": 694,
        "constant": false,
        "quantiles": {
          "0.25": 45.43797003449848,
          "0.5": 68.7891309588895,
          "0.75": 95.11139924123641
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
        "missing_rate": 0.029167,
        "cardinality": 3,
        "constant": false
      },
      "payment_method": {
        "kind": "categorical",
        "missing_rate": 0.030556,
        "cardinality": 2,
        "constant": false
      }
    },
    "positive_rate": 0.36527777777777776,
    "minority_rate": 0.36527777777777776,
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

- 缺失值和类别编码策略 — `churn_business.md`:8（Knowledge:78a1b7dd32ededc8b25a）
  - 图谱路径：requirement:missing_values → strategy:imputation → Knowledge:78a1b7dd32ededc8b25a
  - 图谱路径：requirement:train_only_preprocessing → strategy:train_only_pipeline → Knowledge:78a1b7dd32ededc8b25a
  - 图谱路径：requirement:unknown_categories → strategy:unknown_safe_encoding → Knowledge:78a1b7dd32ededc8b25a
- Missing Value Strategy — `churn_business.md`:8（Knowledge:44b6fc6d2509cc95e94a）
  - 图谱路径：requirement:missing_values → strategy:imputation → Knowledge:44b6fc6d2509cc95e94a
  - 图谱路径：requirement:train_only_preprocessing → strategy:train_only_pipeline → Knowledge:44b6fc6d2509cc95e94a
  - 图谱路径：requirement:unknown_categories → strategy:unknown_safe_encoding → Knowledge:44b6fc6d2509cc95e94a
- Unknown Category Handling — `churn_business.md`:8（Knowledge:4edc002bb0284eb45ffa）
  - 图谱路径：requirement:missing_values → strategy:imputation → Knowledge:4edc002bb0284eb45ffa
  - 图谱路径：requirement:train_only_preprocessing → strategy:train_only_pipeline → Knowledge:4edc002bb0284eb45ffa
  - 图谱路径：requirement:unknown_categories → strategy:unknown_safe_encoding → Knowledge:4edc002bb0284eb45ffa
- Pipeline Requirements — `churn_business.md`:8（Knowledge:9952701fd46b60d14e80）
  - 图谱路径：requirement:missing_values → strategy:imputation → Knowledge:9952701fd46b60d14e80
  - 图谱路径：requirement:train_only_preprocessing → strategy:train_only_pipeline → Knowledge:9952701fd46b60d14e80
  - 图谱路径：requirement:unknown_categories → strategy:unknown_safe_encoding → Knowledge:9952701fd46b60d14e80
- Categorical Encoding — `churn_business.md`:8（Knowledge:be77c229330f64e49f61）
  - 图谱路径：requirement:missing_values → strategy:imputation → Knowledge:be77c229330f64e49f61
  - 图谱路径：requirement:train_only_preprocessing → strategy:train_only_pipeline → Knowledge:be77c229330f64e49f61
  - 图谱路径：requirement:unknown_categories → strategy:unknown_safe_encoding → Knowledge:be77c229330f64e49f61
- 评价指标与业务约束 — `churn_business.md`:14（Knowledge:8c8cd3a7810e4b6a5290）
  - 图谱路径：requirement:class_imbalance → strategy:class_weighting → Knowledge:8c8cd3a7810e4b6a5290

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
