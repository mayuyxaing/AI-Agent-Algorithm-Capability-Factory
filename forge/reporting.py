"""Readable reports and a dependency-free local graph overview."""
import html
from pathlib import Path
from .util import read_json


def report_markdown(report: dict) -> str:
    lines = ["# 算法能力验证报告", "", f"- Run: `{report['run_id']}`",
             f"- 状态：{report['status']}", f"- 模式：{report['mode']} / {report['model']}",
             f"- 故障注入：{report['fault_injection']}",
             f"- 数据 SHA256：`{report['dataset_sha256']}`", "",
             "## 需求", "", report["description"], "", "## 检索依据", ""]
    if report.get("error"):
        lines += ["运行错误：", "", "```text", report["error"], "```", ""]
    ingestion = report.get("knowledge_ingestion")
    if ingestion:
        lines += [f"知识抽取审计：status={ingestion['status']}，LLM 接受 {ingestion['llm_accepted']} 条，拒绝 {ingestion['llm_rejected']} 条；不合格事实未进入能力图谱。", ""]
    for item in report["knowledge"]:
        lines.append(f"- {item['title']} — `{item.get('source_path')}`:{item.get('line')}（{item['id']}）")
    lines += ["", f"历史经验检索数：{len(report['experiences'])}", "", "## 候选比较（验证集）", "",
              "| 算法 | 通过 | 尝试次数 | ROC-AUC | F1 |", "|---|---|---|---|---|"]
    for item in report["candidates"]:
        scores = item["attempts"][-1].get("metrics", {}) if item["attempts"] else {}
        lines.append(f"| {item['algorithm']} | {item['passed']} | {len(item['attempts'])} | {scores.get('roc_auc', 'N/A')} | {scores.get('f1', 'N/A')} |")
        lines += []
    lines += ["", "## 方案依据与修复轨迹", ""]
    for item in report["candidates"]:
        lines.append(f"### {item['algorithm']}")
        lines += ["", item["rationale"], ""]
        for attempt in item["attempts"]:
            lines += [f"- attempt {attempt['attempt']}: passed={attempt['passed']}, injected={attempt['injected']}",
                      f"  - code: `{attempt['code_path']}`; SHA256: `{attempt['code_sha256']}`",
                      f"  - checks: `{attempt.get('checks', {})}`",
                      f"  - explanation: {attempt.get('explanation', '')}"]
            if attempt.get("error"):
                lines += ["", "```text", attempt["error"], "```", ""]
    lines += ["", "## 最终测试集评估", "", f"选定算法：{report.get('selected')}", "",
              "```json", __import__('json').dumps(report.get("test_result"), ensure_ascii=False, indent=2), "```", "",
              "## 解释边界", "",
              "数据为固定种子的模拟数据，指标不能代表真实客户业务效果。按验证集选择，最终测试结果不参与修复。",
              "Mock 的需求解析、规划、代码生成和修复是规则/模板行为，不能作为真实大模型效果证据。",
              "角色按状态机顺序交接；未实现自主并行多智能体，也未实现操作系统级安全沙箱。",
              "python_peak_mb 仅统计 Python 内存分配，非进程总内存；wall_seconds 包含进程启动。", ""]
    return "\n".join(lines)


def graph_html(graph_path: Path, output: Path):
    graph = read_json(graph_path)
    nodes = graph["nodes"]
    rows = "".join(f"<tr><td>{html.escape(n['kind'])}</td><td>{html.escape(n['title'])}</td><td><code>{html.escape(n['id'])}</code></td></tr>" for n in nodes)
    # Inline SVG permits offline inspection without CDN JavaScript.
    import math
    positions = {n["id"]: (500 + 390 * math.cos(2 * math.pi * i / max(1, len(nodes))),
                            430 + 350 * math.sin(2 * math.pi * i / max(1, len(nodes)))) for i, n in enumerate(nodes)}
    svg = []
    for edge in graph["edges"]:
        x1, y1 = positions[edge["source"]]
        x2, y2 = positions[edge["target"]]
        svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#a8bacb"><title>{html.escape(edge["relation"])}</title></line>')
    for node in nodes:
        x, y = positions[node["id"]]
        label = html.escape(node["title"])
        svg.append(f'<g><circle cx="{x}" cy="{y}" r="6" fill="#176b80"><title>{html.escape(node["id"])}: {label}</title></circle><text x="{x+8}" y="{y}" font-size="10">{label[:45]}</text></g>')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>能力知识图谱</title><style>body{font-family:system-ui;max-width:1200px;margin:32px auto;color:#17354b}td,th{padding:8px;border-bottom:1px solid #ddd}code{font-size:11px}svg{width:100%;background:#f4f8fa}table{width:100%}</style><h1>算法能力知识图谱</h1><p>离线关系图；悬停查看节点与关系。JSON 是完整属性数据源。</p><svg viewBox="0 0 1150 880">' + ''.join(svg) + '</svg><table><tr><th>类型</th><th>标题</th><th>ID</th></tr>' + rows + '</table></html>', encoding="utf-8")
