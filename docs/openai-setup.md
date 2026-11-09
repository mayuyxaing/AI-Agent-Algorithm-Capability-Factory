# OpenAI API 本机接入

ChatGPT 网页用于聊天；项目自动调用使用 API 平台的项目密钥和 API 可用额度。不要把 https://chatgpt.com/ 作为 API 地址。

## 账号准备

1. 登录 [API 平台](https://platform.openai.com/)，选择或创建自己的项目。
2. 在 [API Keys](https://platform.openai.com/api-keys) 创建项目密钥并妥善保管；不要发送到聊天、截图、代码仓库。
3. 打开 [API Billing](https://platform.openai.com/settings/organization/billing/overview) 检查可用额度和计费状态。若账号显示可领取的订阅 API 额度，先按页面要求领取，再核对到账；否则按需配置付费额度。不要仅凭 ChatGPT 能聊天就判断 API 已可用。

参考：[官方快速开始](https://developers.openai.com/api/docs/quickstart)、[符合条件的订阅 API 额度说明](https://developers.openai.com/api/docs/guides/chatgpt-api-credits)。是否具有资格、可用模型与实际额度以账号页面为准。

## 隐藏输入密钥并检查连接

按 README 安装依赖和项目后，在仓库根目录的个人交互终端运行：

```powershell
python scripts/run_openai.py
```

粘贴密钥并回车。字符不显示是正常现象。脚本只在当前进程设置 OPENAI_API_KEY，不写密钥文件，退出时移除临时设置。

这一步会发送一次真实需求解析请求，因此会使用 API 额度。成功应显示 provider=openai、probe=true、task=churn_classification。默认模型 gpt-4.1-mini，可用 `--model YOUR_MODEL` 替换，实际访问权限由账号决定。

## 跑完整闭环

```powershell
python scripts/run_openai.py --run
```

再次隐藏输入密钥。脚本建立独立的 runtime/openai/knowledge.sqlite，先从材料抽取知识，再进行需求解析、规划、生成、验证与修复。报告保存在 runtime/openai/runs/，不会混用 mock 演示的故障经验。

文档、需求和检索内容会发送到官方 OpenAI API；示例材料全为模拟内容。不发送完整数据表，训练和评估在本地运行。调用记录含 response_id、服务端 usage 与耗时，不含认证头。store=false 不等于对所有服务端日志/保留策略作保证。

已有环境变量时，也可以直接运行：

```powershell
python -m forge doctor --provider openai
python -m forge doctor --provider openai --probe
python -m forge run --task-file examples/task.txt --provider openai --model gpt-4.1-mini --output runtime/openai/runs
```

doctor 不加 --probe 只检查本地配置，不证明密钥有效，也不发请求。完整运行的调用次数取决于修复次数；每个请求输出上限默认 6000 tokens，修复预算见任务配置。这些限制不是账户消费的硬上限。

## 常见失败

| 提示 | 排查方向 |
|---|---|
| OPENAI_API_KEY is not configured | 在个人终端使用隐藏输入脚本 |
| HTTP 401 | 密钥无效、撤销或复制不完整 |
| HTTP 403/404 | 项目权限、模型权限、服务可用性或模型名 |
| HTTP 429 | 查看账户可用额度及请求速率限制 |
| connection failed or timed out | 网络连通性；不等于算法生成失败 |
| incomplete | 输出上限或模型响应未完成；不记作验证通过 |

服务连接/认证失败会停止工作流并保留失败报告，不触发无意义的代码修复，也不回退到 mock。

## 实现与实测边界

使用 [Responses API 文本生成](https://developers.openai.com/api/docs/guides/text)，以 JSON 文本承载结构化交接；宿主另行检查 JSON 和任务契约。不是严格 JSON Schema 约束生成。标准库 HTTP 调用，无新增第三方依赖。

接口契约测试使用虚假密钥和模拟响应，不消耗额度。本交付尚无真实 OpenAI 模型调用结果；只有配置账号后实际成功的运行才可在答辩中作为真实模型证据。
