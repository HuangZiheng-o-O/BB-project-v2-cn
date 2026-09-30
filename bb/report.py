"""确定性地生成"问题 + 回答 + 来源证据"的 Markdown 报告。"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from bb.models import Anchor
from bb.source import Corpus


def model_call_count(trace: list[dict[str, Any]]) -> int:
    """统计一个阶段轨迹中的模型调用次数;工具执行条目没有 usage 字段,不计入。"""
    return sum("usage" in entry for entry in trace)


def _cited_lines(citations: list[str], corpus: Corpus) -> dict[str, set[int]]:
    """把 "SRC:L1,L3-L5" 形式的引用解析并校验,按来源汇总为 {来源ID: 行号集合}。"""
    grouped: dict[str, set[int]] = defaultdict(set)
    for reference in citations:
        # 以第一个冒号分隔来源 ID 与行号部分
        source_id, separator, suffix = reference.partition(":")
        if not separator:
            raise ValueError(f"Invalid source citation: {reference}")
        numbers: list[int] = []
        for part in suffix.split(","):
            # 支持单行 L12 和区间 L12-L15
            match = re.fullmatch(r"L(\d+)(?:[-–]L(\d+))?", part)
            if not match:
                raise ValueError(f"Invalid source citation: {reference}")
            first = int(match.group(1))
            last = int(match.group(2)) if match.group(2) else first
            if last < first:
                raise ValueError(f"Invalid source citation: {reference}")
            numbers.extend(range(first, last + 1))
        # 确认来源存在且行号未越界
        corpus.validate_anchor(Anchor(source_id=source_id, lines=numbers))
        grouped[source_id].update(numbers)
    return grouped


def render_answer_markdown(
    question: str,
    answer: str,
    citations: list[str],
    corpus: Corpus,
    online_model_calls: int,
    citation_audit: list[str] | None = None,
) -> str:
    """在本地复制被引用的原文行生成报告,不再额外调用模型。"""
    sections = [
        "# Question", "", question.strip(), "",
        "# Answer", "", answer.strip(), "",
        "# Evidence", "",
        "Original document lines referenced in the answer:", "",
    ]
    grouped = _cited_lines(citations, corpus)
    if not grouped:
        sections.extend(["No validated source lines were cited.", ""])
    # 每个来源文档一个小节,列出被引用的原文行(行号补零到 4 位)
    for source_id, numbers in sorted(grouped.items()):
        source = corpus.get(source_id)
        sections.extend([f"## {source_id} — [{source.filename}](<{source.absolute_path}>)", "", "```text"])
        sections.extend(f"L{number:04d} {source.lines[number - 1]}" for number in sorted(numbers))
        sections.extend(["```", ""])
    # 若引用审计仍有警告,附在报告中
    if citation_audit:
        sections.extend(["# Citation audit warnings", ""])
        sections.extend(f"- {warning}" for warning in citation_audit)
        sections.append("")
    sections.extend(["# Run information", "", f"- Online model calls: {online_model_calls}", ""])
    return "\n".join(sections)
