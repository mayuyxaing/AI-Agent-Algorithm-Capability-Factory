# 运行与案例说明

我准备了客户流失流程与修复演示，以及真实文本代码的半自动接口修复复现。真实 GLM 历史报告保存在 examples 中。

## 环境配置

在解压后的 capability-forge 目录执行，推荐 Python 3.12：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

以下 Windows 命令直接使用虚拟环境解释器。Linux/macOS 将路径替换为 `.venv/bin/python`。

## 案例一：客户流失预测

输入需求保存在 [task.txt](../examples/task.txt)。材料包含客户字段、缺失值处理、未知类别编码及评价要求。

```powershell
.\.venv\Scripts\python.exe -m forge init
.\.venv\Scripts\python.exe -m forge run --task-file examples/task.txt --provider mock --output runtime/demo-churn
.\.venv\Scripts\python.exe -m forge run --task-file examples/task.txt --provider mock --inject-fault --output runtime/demo-churn-repair
.\.venv\Scripts\python.exe -m forge graph --output runtime/demo-graph.json
```

第一条 run 演示生成、候选比较、最终测试与回写；第二条演示显式故障后的有限修复；graph 导出 JSON 和同名 HTML。上述命令使用 mock，重点是本地流程复现。

真实 GLM 客户流失案例：

- [完整报告](../examples/v2/real-success/report.md)
- [逐步记录](../examples/v2/real-success/events.json)
- [模型原始代码](../examples/v2/real-success/logistic/attempt_0/model_response.py)
- [宿主注入故障后的代码](../examples/v2/real-success/logistic/attempt_0/algorithm.py)
- [GLM 修复代码](../examples/v2/real-success/logistic/attempt_1/algorithm.py)

该真实案例选中逻辑回归，测试 ROC-AUC=0.7829、F1=0.6432。未知类别故障由宿主显式注入，修复响应来自真实 GLM。

## 案例二：工单文本分类

输入需求保存在 [text_task.txt](../examples/text_task.txt)，数据为带缺失值和标签噪声的模拟工单。我把文本输入、字符 TF-IDF 和模型检查放在任务插件中，复用同一套工作流。

```powershell
.\.venv\Scripts\python.exe scripts/assisted_text_example.py --source-run examples/v3/text-assisted-repair/source-run --output runtime/demo-text
```

这条命令读取归档的真实 GLM 代码，恢复图谱，应用固定 DataFrame 接口补丁，再重新训练、比较候选和评估测试集；不需要 API Key。输出须为新目录，重复复现时可更换名称。

归档结果见 [文本报告](../examples/v3/text-assisted-repair/runs/20261115T045921Z-76a2bd42/report.md)。两候选验证 F1 均为 0.8221，按配置与候选顺序选择逻辑回归。测试 ROC-AUC=0.8701、F1=0.8765，12 项检查通过。

我在开发阶段借助编码助手完成输入接口修正，该案例记录为 `assisted_glm`。原始自主失败、修改差异和修正后的代码均保留。

## 输出与预测

每个运行目录包含 `report.md`、`report.json`、`events.json`、图谱 JSON/HTML，以及各候选的 algorithm.py、模型和验证结果。最终选中版本位于 selected。

使用本次重新训练的文本模型预测：

```powershell
$textRun = Get-ChildItem runtime/demo-text/runs -Directory | Sort-Object Name -Descending | Select-Object -First 1
.\.venv\Scripts\python.exe scripts/predict.py --task text_classification --artifact "$($textRun.FullName)/selected" --input examples/text_predict_input.csv --output runtime/text-predictions.csv
```

输出新增 `urgent_probability` 和 `predicted_label`，三条工单的[预测结果样例](../examples/v3/text-assisted-repair/predictions.csv)已一并提供。

## 自动测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

已归档 [148 项测试通过记录](../examples/test-results.txt)。精确环境保存在报告 environment 字段中。以上复现命令会重新训练本地模型。

## 真实调用与版本历史

安装依赖后，可双击 `check-glm.cmd` 检查连接、`run-glm.cmd` 执行客户流失、`repair-glm.cmd` 执行显式修复演示、`text-glm.cmd` 执行文本任务。密钥在终端隐藏输入，不保存在项目中。平台权限与拥堵会影响实时调用，历史报告和离线案例可独立查看。

提交包含 `git-history.bundle`，可恢复分阶段提交：

```powershell
git clone git-history.bundle ../capability-forge-history
```
