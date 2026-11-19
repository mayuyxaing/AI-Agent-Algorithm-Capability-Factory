# 算法能力验证报告

- Run: `20261118T140013Z-2d6af97e`
- 状态：failed
- 模式：glm / glm-4.7-flash
- 故障注入：False
- 数据 SHA256：`e2b64050532ce82160943ac3e6cbb0504a4526df7c04b4cd4af2bc29ad0c796a`

## 需求

客户流失预测，处理缺失值、未知类别和类别不平衡，比较逻辑回归和随机森林，使用 ROC-AUC 和 F1 评价。

## 检索依据

运行错误：

```text
LLMProviderError: BigModel HTTP 429 [business_code=1305]: 模型访问量过大，请稍后重试
```

知识抽取审计：status=passed，LLM 接受 0 条，拒绝 0 条；不合格事实未进入能力图谱。

工作流 v3；实验组=knowledge；每候选生成预算=3；未向真实模型提供完整算法模板。

## 训练数据画像与实现计划

```json
{
  "data_profile": {
    "scope": "training_only",
    "rows": 360,
    "columns": {
      "tenure": {
        "kind": "numeric",
        "missing_rate": 0.294444,
        "cardinality": 71,
        "constant": false,
        "quantiles": {
          "0.25": 21.0,
          "0.5": 39.0,
          "0.75": 58.0
        }
      },
      "monthly_charges": {
        "kind": "numeric",
        "missing_rate": 0.261111,
        "cardinality": 266,
        "constant": false,
        "quantiles": {
          "0.25": 45.00381034562399,
          "0.5": 64.5098455878981,
          "0.75": 95.61564553859307
        }
      },
      "support_calls": {
        "kind": "numeric",
        "missing_rate": 0.258333,
        "cardinality": 7,
        "constant": false,
        "quantiles": {
          "0.25": 0.5,
          "0.5": 1.0,
          "0.75": 2.0
        }
      },
      "contract": {
        "kind": "categorical",
        "missing_rate": 0.247222,
        "cardinality": 3,
        "constant": false
      },
      "payment_method": {
        "kind": "categorical",
        "missing_rate": 0.3,
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
