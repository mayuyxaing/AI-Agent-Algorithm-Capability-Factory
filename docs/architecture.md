# 架构、接口与可信边界

## 为什么采用这些组件

- Python + scikit-learn：小数据二分类训练速度快，Pipeline 把预处理与模型绑定，便于验证数据划分。
- SQLite：知识、关系和报告本地持久化，无需部署服务，关系外键和 SQL 容易审查。
- 显式 Python 状态机：每个角色有结构化输入输出，调试与测试直接；避免为一周原型引入复杂 Agent 框架。
- Ollama HTTP：可接本地开源模型，无需提交密钥；模型由使用者指定，不假定某个模型已安装。
- Mock + 模板：离线演示可复现，和真实 LLM 模式分开记录，不混淆效果。
- CLI：满足提交需求，便于老师在终端复现，HTML 只用于离线图谱概览。

## 角色交接

1. understand 接收自然语言，输出 task/target/metrics/constraints；当前只接受客户流失二分类。
2. retrieve 查询文档知识、函数和历史失败案例；每条保留来源与引文。
3. plan 提交知识及可用插件，输出候选 ID、理由和证据 ID。宿主拒绝未知/重复算法和伪造引用。
4. generate 返回完整 Python 文件；每次保留 prompt 上下文、响应和代码摘要。
5. validate 宿主静态检查后启动子进程，训练并执行接口检查；宿主从预测计算指标。
6. repair 接收上一版代码及结构化反馈，最多按配置修复两轮；生成代码逐版保留。
7. final_evaluation 按验证集指标选出通过者，对选中代码进行一次测试集评估。失败即报告 test_failed，不能回流调整。
8. persist 保存运行、版本、指标、环境和失败经验，导出 JSON 与 HTML。

角色共享同一 LLM 适配器，串行调度；这是结构化多角色工作流，不声称实现自主并行多智能体。events.json 包含每一步的输入输出，report.json 是汇总结果。

## 生成算法契约

```python
def build_model(seed=42):
    # 返回未拟合的 sklearn.pipeline.Pipeline
    ...

def predict(model, frame):
    # 返回 shape=(len(frame),) 的正类概率，范围 [0,1]
    # 缺少必要字段时抛 ValueError；保持输入行顺序
    ...
```

宿主控制数据：分层划分 train/validation/test=60/20/20。子进程只获得训练标签和待预测特征，不获得验证/测试标签。接口测试覆盖概率形状、范围、未知类别、全缺失值、字段缺失、列顺序、重复预测和模型保存加载。accuracy/F1 阈值固定 0.5，不调阈值。基线为训练集正例比例的常数预测。

需求中的 metrics/constraints 会保留供规划使用，但可执行验收策略以 configs/churn.json 为准；本版不把任意自然语言门槛自动转成配置，也不支持任意字段/目标。遇到其他任务应拒绝或先扩展验证器。

## 可扩展点

- 候选算法：添加 forge/plugins/*.json 的 import、expression、rationale，mock 模板即可生成新候选；插件是受信任项目代码，仍须通过 AST 检查和统一验证。
- 指标：在 validation.METRICS 注册高者更优的函数，再加入配置；若是低者更优指标，须扩展排序方向。
- 任务：新增字段 schema、数据读取、模板、接口检查、配置任务分派；当前只提供一个任务类型，没有宣称完成跨场景迁移。
- LLM：实现 mode、model、call(role, context)；真实模型输出必须通过宿主校验。

## 安全与限制

AST import allowlist、敏感环境变量名称过滤、独立工作目录、python -I、执行超时和线程限制是防误操作措施。它们不能阻止恶意 Python 绕过检查，不能阻止读取其他本地文件，不能提供网络隔离；输出 capture 也没有强制字节上限。只运行受信任环境中生成并审阅的代码。部署到公共服务前必须使用容器/VM、禁止网络、限制文件系统/进程/内存并限制输出。

训练 Pipeline 结构被检查，正常模板只在训练集拟合；无法靠当前静态检查形式化证明任意生成代码无泄漏。pickle/joblib 文件仅可从可信来源加载。python_peak_mb 为 tracemalloc 统计值，不包含全部原生库内存，不应标注为进程峰值 RSS。

重复运行同一固定数据集用于展示工程流程，不能作为独立样本实验；没有证明知识增强或多角色工作流带来统计显著提升。历史失败修复建议目前是保守规则，尚未实现自动根因推断；mock 修复直接再生安全模板。真实模型能力需要在配置 Ollama 后另行评测。

## 技术参考

- [Ollama chat API](https://docs.ollama.com/api/chat)：POST /api/chat，stream=false，format=json。
- [scikit-learn 常见陷阱](https://scikit-learn.org/stable/common_pitfalls.html)：先划分数据、在训练集拟合预处理并使用 Pipeline。
