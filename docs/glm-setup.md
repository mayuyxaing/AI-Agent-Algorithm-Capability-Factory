# 智谱 GLM-4.7-Flash 接入

本项目按用户选择固定使用 `glm-4.7-flash`。官方发布说明提供免费 API 调用，账户权限和速率限制以控制台为准；程序不切换 `glm-4.7`、`glm-4.7-flashx` 或其他模型，也不回退到 OpenAI。

## 先准备智谱密钥

登录 [智谱开放平台](https://bigmodel.cn/)，在控制台查看 GLM-4.7-Flash 是否可调用且标为免费，然后到 API Keys 创建密钥。这里需要智谱自己的密钥，不能使用 OpenAI 的密钥。密钥只在自己的终端输入，不发到聊天、截图或代码仓库。

## Windows 双击运行

1. 双击根目录 `check-glm.cmd`。
2. 按提示粘贴智谱 API Key，回车；字符不显示是正常现象。
3. 成功结果包含 provider=glm、model=glm-4.7-flash、probe=true、task=churn_classification。不含密钥的诊断保存到 `runtime/glm/connection-check.json`。
4. 连接成功后双击 `run-glm.cmd`，再次输入密钥。完整流程先做知识抽取，再做规划、生成、验证与修复。

脚本优先使用仓库 `.venv`，其次使用本地开发环境 `work/.venv`，最后尝试系统 Python。新电脑需要先按 README 安装项目。密钥仅设置在当前 Python 进程的 ZHIPU_API_KEY 中，临时设置在退出时移除，不写密钥文件。

真实模型数据库为 `runtime/glm/knowledge.sqlite`，报告在 `runtime/glm/runs/`，不会混入 mock 或 OpenAI 的历史示例。该目录默认不提交 Git。再次启动完整流程时，若材料摘要、模型和抽取审计匹配，将复用已核对知识，不重复抽取；可用 --refresh-knowledge 强制重做。

成功的需求理解、规划及代码响应保存到 `runtime/glm/checkpoints/`。只有模型、数据、策略、种子、提示词和该步骤完整输入一致才复用，并记录来源 run_id。验证、指标和测试结果始终重新计算，不使用缓存。生成代码响应被缓存不代表代码已经通过验证；仍需走全部检查与修复。不同任务上下文或材料改变会使相关缓存失效。

## 命令行

```powershell
python scripts/run_glm.py        # 隐藏输入密钥，只做一次真实连接测试
python scripts/run_glm.py --run  # 真实模型完整流程
```

如果已在当前终端设置 ZHIPU_API_KEY：

```powershell
python -m forge doctor --provider glm --probe --output runtime/glm/connection-check.json
python -m forge --db runtime/glm/knowledge.sqlite init
python -m forge --db runtime/glm/knowledge.sqlite ingest materials --provider glm
python -m forge --db runtime/glm/knowledge.sqlite run --task-file examples/task.txt --provider glm --output runtime/glm/runs
```

不加 --probe 的 doctor 只检查本地配置，不证明密钥有效。

## 请求与错误处理

使用智谱官方 `https://open.bigmodel.cn/api/paas/v4/chat/completions`，Bearer 认证，非流式 JSON 输出。默认关闭 thinking，最大输出 8192 tokens，单次请求超时 180 秒。所有调用串行。仅对模型拥堵业务码 1305 重试，通用 CLI 默认最多三次请求；Windows 完整运行入口最多六次，依次等待 10、20、45、60、60 秒。仍失败则保存已有结果并停止，下次运行可复用成功步骤。认证错误、欠费、账号速率限制等不自动重试。重试事件进入调用记录，不切换其他模型。

- 缺少 ZHIPU_API_KEY：从 check-glm.cmd 在本机输入。
- HTTP 401：核对智谱密钥是否有效。
- HTTP 403/404：核对免费模型的账号权限与服务可用性。
- HTTP 429：核对免费模型并发/速率限制，稍后重试。
- 错误信息中的 business_code 提供更精确的原因：1000 身份验证失败、1003 令牌过期、1005 二次认证保护、1113 账号欠费、1302 速率限制。程序只输出已识别的业务码和本地提示，不回显远端原始错误正文。不要仅凭 HTTP 401 就判断为余额不足。
- response incomplete：输出截断或过滤，不作为成功结果。
- JSON/原文引证核对失败：模型产物未满足接口要求；保留现有证据，不冒充验证通过。

知识抽取通过 source_chunk_id 引用编号原文片段，由宿主附加引证。单条无效来源会被拒绝并写入 `runtime/glm/ingestion-report.json`，默认继续使用通过核对的 LLM 事实及确定性抽取知识；控制台显示 llm_accepted、llm_rejected 与 partial_llm 状态。即使所有 LLM 事实被拒绝，也不会伪造 LLM 抽取成功。需要严格失败策略时，用 ingest --strict-extraction。算法报告保留最近一次知识抽取审计。

HTTP 错误不输出远端原始错误体，避免回显敏感内容。连接或认证失败不触发算法代码修复，不回退到 mock。模型固定并不能保证未来平台价格不变，实际使用前应核对控制台标价。

## 证据边界

单元测试使用虚假密钥与模拟 HTTP 响应，不调用真实账户。另已完成一次真实 GLM 完整流程，证据位于 examples/glm/successful-run/，摘要见 examples/glm/PROGRESS.md。mock 和真实模型的报告分开保存；相同数据和参考模板导致指标一致，不能据此证明真实模型优于 mock。真实样例没有触发 repair，且需求 constraints 出现占位值，报告保留原样并列明限制。

参考：[模型调用文档](https://docs.bigmodel.cn/cn/guide/models/free/glm-4.7-flash)、[免费发布说明](https://www.zhipuai.cn/zh/news/148)、[对话补全 API](https://docs.bigmodel.cn/api-reference/模型-api/对话补全)、[思考模式](https://docs.bigmodel.cn/cn/guide/capabilities/thinking-mode)。
