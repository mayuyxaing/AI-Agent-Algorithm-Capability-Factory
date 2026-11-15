# 多任务与插件扩展

已接入 churn_classification（表格客户流失）与 text_classification（客服工单紧急程度）。共用工作流、执行器、概率接口、指标、候选比较、修复调度、报告和知识回写；字段、画像、计划校验、预处理检查及边界用例由任务插件提供。

## 证据与边界

原真实 GLM 森林代码在加载器修复后重新执行验证集检查，全部通过，包括自定义类保存加载，见 examples/v3/loader-recheck/recheck.json。源码摘要未变，原失败报告保留；未重新选择模型或调优测试集。

文本离线闭环已运行：字符 TF-IDF + LogisticRegression / MultinomialNB，含缺失/空文本、未见词表、接口及模型重载检查。源数据为带随机标签噪声的模拟工单，相似模板较多，不能证明真实业务泛化效果。文本真实 GLM 和真实三组对照尚未运行，现有真实 API 证据属于客户流失任务。

## 运行第二场景

```powershell
python -m forge --db runtime/text.sqlite init --task text_classification
python -m forge --db runtime/text.sqlite run --task text_classification --task-file examples/text_task.txt
```

真实模型：双击 text-glm.cmd，或运行 `python scripts/run_glm.py --run --task text_classification`，在自己的终端隐藏输入智谱密钥。仍固定 glm-4.7-flash；发送需求、训练统计与知识，不发送完整数据表。对照与未知类别故障演示目前只针对客户流失。

## 任务契约

forge.tasks 提供 register_task/get_task。

- 属性：id/title/target/features/numeric/categorical/interface/config/data。
- generate/load_split/profile：数据准备、划分和训练集统计。
- validate_spec/validate_plan/catalog/schema：需求与计划空间。
- check_preprocessing/variants：实际计划一致性与场景边界用例。
- 非内置任务另提供 prompt(role)/mock_call(role,context)，用于真实提示与明确标注的离线生成。

受信任扩展包可声明安装入口，宿主和子进程从同一 Python 环境发现：

```toml
[project.entry-points."capability_forge.tasks"]
my_task = "my_package:TaskPlugin"
```

TaskPlugin 为无参类/工厂。公共执行契约目前仍是二分类概率和 sklearn Pipeline，回归/推荐等输出需要扩展公共契约。

## 算法与指标插件

算法描述集中在 forge/plugins/*.json：tasks、id、name、estimator、import、defaults、parameter_space、seeded、fixed_parameters。规划参数和执行时分类器检查从描述读取。文本 mock 使用通用 TF-IDF 模板构建兼容的概率分类器。

安装式算法入口组为 capability_forge.algorithms，无参工厂返回一个描述字典或字典列表。分类器需要 predict_proba，适配任务特征和模板；新类型预处理仍要实现任务检查，不能宣称任意模型仅写 JSON 就可运行。插件是受信任代码，子进程不是 OS 沙箱。

register_metric(name,scorer) 注册接收 `(y_true,positive_probabilities)` 的函数；安装入口组 capability_forge.metrics 的工厂返回 `{name:scorer}`。配置可选择注册指标。新增 balanced_accuracy 已用于文本任务。当前只支持高者更优、minimum 门槛和 [0,1] 分类指标，低者更优或无界指标需另扩展方向/门槛契约。

## 失败经验

规则归因为 validator_loader、unknown_category、plan_noncompliance、text_dataframe_contract、resource_limit、serialization、unclassified，记录 category/confidence/remedy/reusable。文本输入形状错误提供提取 `X['text']` 的针对性建议；当前轮修复和后续经验检索均可使用。加载器故障和未知原因不作为算法修复经验复用，检索按 task 隔离。这是规则假设，不是通用 LLM 根因推断；原始 traceback 和版本均保留。

## 自动接口目录

```powershell
python -m forge catalog --output docs/generated-interfaces.md
```

文档来自实际注册表，包含任务、接口、参数空间、算法和指标。这是 CLI/插件契约目录，没有宣称生成 HTTP OpenAPI 文档。
