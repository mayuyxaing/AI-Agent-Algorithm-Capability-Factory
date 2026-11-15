# 工作流 v2 真实 GLM 结果

实际运行：20261115T024040Z-c6fa258d，glm-4.7-flash，status=passed，选中 logistic。

- [原始报告](real-success/report.md) / [JSON](real-success/report.json)
- [完整角色交接](real-success/events.json)
- [最终代码](real-success/selected/algorithm.py)
- [逻辑回归初版原始模型输出](real-success/logistic/attempt_0/model_response.py)
- [注入缺陷后的实际执行版本](real-success/logistic/attempt_0/algorithm.py)
- [真实模型修复后的版本](real-success/logistic/attempt_1/algorithm.py)

本次不提供完整参考实现，提供接口、训练画像、检索知识和结构化计划。需求理解复用了之前的真实模型响应；规划在一次格式纠正后通过。逻辑回归的未知类别故障由宿主显式注入，真实 GLM 根据验证反馈完成修复。本例不是自然生成错误的修复实验。

最终测试 ROC-AUC=0.7829，F1=0.6432，accuracy=0.7042，12 项检查全部通过。数据为模拟数据，不能代表实际业务效果。候选按验证集比较，测试结果未用于修复。

随机森林候选在三次尝试中均未通过模型重载检查，未参与最终选择。原代码含自定义 FeatureValidator；验证加载器没有将动态模块注册到 sys.modules，导致 PicklingError。修复响应的说明与代码也有不一致之处（声称删除类但代码摘要未改变）。保留全部原始失败，不将其改标为通过。

随后已修正 worker 和预测脚本的模块注册；当前代理环境无法启动原 Python 3.14 解释器，因此该加载器修正尚未对随机森林重新运行机器学习验证。它不影响原始逻辑回归已通过的结果。不能将上述森林失败全部归因为模型能力，也不能声称两个候选均已通过。

此时仍没有真实 GLM 三组对照结果，不能推断知识增强带来统计提升。
