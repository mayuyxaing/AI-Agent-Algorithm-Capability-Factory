# 演示操作顺序

我先展示已经归档的真实结果，再运行不依赖外部接口的本地流程。现场真实 API 的可用性与离线复现分别说明。

## 演示前准备

在项目根目录安装环境，具体步骤见 [运行说明](run-and-results.md)。Windows 使用 `.venv\Scripts\python.exe`，无需激活环境。打开以下文件：

1. [正式实验报告](../examples/real-benchmark/FULL-RESULTS.md)：27 单元的通过率、失败原因和成本。
2. [代码讲解提纲](code-walkthrough.md)：从入口、规划、候选执行讲到最终测试与回写。
3. [最新演示讲稿](../docs/demo.md)：真实修复轨迹及结果文件入口。

## 本地运行

先使用一个未使用过的演示目录名，例如下面的 `runtime/presentation-01`；重复演示时改为 `presentation-02`，不删除旧结果。

```cmd
.venv\Scripts\python.exe -m forge --db runtime/presentation-01/knowledge.sqlite init
.venv\Scripts\python.exe -m forge --db runtime/presentation-01/knowledge.sqlite run --task-file examples/task.txt --provider mock --inject-fault --output runtime/presentation-01/runs
.venv\Scripts\python.exe -m forge --db runtime/presentation-01/knowledge.sqlite graph --output runtime/presentation-01/graph.json
```

我明确说明这里是 mock 离线流程，展示错误检测、有限修复和回写。真实模型证据使用归档的报告与事件，不把离线模板生成当作新的 API 运行。

运行完成后，打开 `runtime/presentation-01/runs/` 下本次目录的 report.md，查看初版失败、修复后检查和最终指标，再打开 graph.html 查看运行与版本关系。

文本半自动案例可另外运行，输出也使用新目录：

```cmd
.venv\Scripts\python.exe scripts/assisted_text_example.py --source-run examples/v3/text-assisted-repair/source-run --output runtime/presentation-text-01
```

这一步复用归档模型源码和明确标注的接口补丁，不产生真实 API 调用。

## 不调用模型地复核正式实验

```cmd
.venv\Scripts\python.exe scripts/summarize_benchmark.py examples/real-benchmark/full-v4 --output runtime/presentation-analysis.json
```

程序核对全部 27 个当前报告的摘要、数据身份及运行状态，并重新计算统计。应得到三组 4/9、8/9、4/9。失败结果保留在分母中，不为演示修改门槛或重选失败单元。

本轮演示命令的实际核对结果见 [presentation-check.json](../examples/presentation-check.json)。
