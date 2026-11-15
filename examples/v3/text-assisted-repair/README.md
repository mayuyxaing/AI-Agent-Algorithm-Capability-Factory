# 真实文本代码的辅助修复与复现

[完整验证报告](runs/20261115T045921Z-76a2bd42/report.md) / [JSON](runs/20261115T045921Z-76a2bd42/report.json) / [修改来源](provenance.json)。

原始 GLM 运行 `20261115T044624Z-10c203ec` 状态为 `no_valid_candidate`，原报告、响应、图谱和各轮代码在 [source-run](source-run/report.md) 原样保留。清洗器没有处理宿主传入的 DataFrame，导致样本数量错误或无法构造 Series。

编码助手在工作流外对两个候选的最后一份代码添加同一接口修正：若 X 是 DataFrame，检查并提取 `X['text']`，然后执行原有清洗逻辑。没有替换算法、修改超参数或降低验证门槛。逐行差异在 [逻辑回归补丁](sources/text_logistic/interface.patch) 和 [朴素贝叶斯补丁](sources/text_nb/interface.patch)。

| 候选 | 验证 ROC-AUC | 验证 F1 | 验证通过 |
|---|---|---|---|
| text_logistic | 0.8455 | 0.8221 | 是 |
| text_nb | 0.8369 | 0.8221 | 是 |

同一工作流根据验证集选择 text_logistic，再执行最终测试：ROC-AUC=0.8701，F1=0.8765，balanced_accuracy=0.8750，12 项接口、稳定性、持久化和指标检查全部通过。验证结果与版本已回写至本次图谱。

这是一份明确标注的半自动复现，模式为 `assisted_glm`，使用已记录的真实模型代码和显式助手补丁；新增 API 调用为 0。它不是 GLM 在无人辅助下自主修复成功，也不改变原运行的失败结论。模拟文本数据上的结果不代表真实业务泛化能力。

在项目根目录安装依赖后，无需 Key 即可复现（使用新的输出目录）：

```powershell
python scripts/assisted_text_example.py --source-run examples/v3/text-assisted-repair/source-run --output runtime/text-assisted-demo
```

脚本核对原数据摘要，从归档图谱恢复知识，应用固定接口补丁，重新训练/验证两个候选，并对选定方案执行同一测试协议。测试集不参与补丁、调参或方案选择。重复复现不构成新的独立实验。
