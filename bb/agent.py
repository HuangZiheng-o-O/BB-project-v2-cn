"""在完整审查快照和来源工具之上运行的、有预算上限的调查型智能体。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from bb.model_provider import ModelPort
from bb.models import Anchor, ReviewSnapshot
from bb.source import Corpus


# 智能体的系统提示词(发给模型的原文,保持英文不变,以免影响模型行为)
AGENT_SYSTEM = """You answer questions about an auditable clinical record review. You are an investigator with tools, not a free-form summarizer.

Use calculate for complete numerical aggregates, scan for complete enumerations, related for an encounter's competing source claims, and search/open for original language. Search is ranked and incomplete; never infer a count or absence from search hits. Follow evidence into source lines before relying on a clinical or conflict claim. Choose tools as needed, within the tool budget.

The snapshot contains candidate observations and decisions. It may omit a concept needed by a new question; investigate original documents using search/open and scan. Distinguish a source statement from a reconciled conclusion. Preserve conflicting values as ranges and identify the missing evidence. If totals_complete is false, reported numeric maxima are known-event subtotals, not global upper bounds. Never turn a planned, scheduled, posted, copied, or unsigned record into delivered treatment. Cite decisive source lines as [SOURCE_ID:L12] or [SOURCE_ID:L12,L13]. Explain inclusion/exclusion and arithmetic for numeric answers. If a claim cannot be established, say so. Do not invent citations or claims. Write the final answer in English. Tool arguments are JSON."""


# 提供给模型的工具声明(名称、说明、JSON Schema 参数)
TOOL_SPECS: list[dict[str, Any]] = [
    {
        # 词法检索:按相关度排序返回来源行,只能用于找线索,不能用于穷举计数
        "name": "search",
        "description": "Rank source lines by lexical relevance. Useful for finding candidates, never for exhaustive counts.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["query"],
        },
    },
    {
        # 打开原始文档:读取某个证据附近的带行号原文
        "name": "open_source",
        "description": "Read original numbered source lines around an evidence claim.",
        "parameters": {
            "type": "object",
            "properties": {
                "source_id": {"type": "string"},
                "first": {"type": "integer"},
                "last": {"type": "integer"},
            },
            "required": ["source_id"],
        },
    },
    {
        # 关联查看:查看一个已对账事件及其所有来源提及
        "name": "related",
        "description": "Inspect one reconciled encounter and every source mention linked to it.",
        "parameters": {
            "type": "object",
            "properties": {"event_id": {"type": "string"}},
            "required": ["event_id"],
        },
    },
    {
        # 全量扫描:分页遍历来源/事件/目标/量表/观察的完整清单
        "name": "scan",
        "description": "Page through the complete inventory of sources, events, goals, measures, or observations.",
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["sources", "events", "goals", "measures", "observations"]},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
                "date": {"type": "string"},
                "service_type": {"type": "string"},
                "theme": {"type": "string"},
            },
            "required": ["kind"],
        },
    },
    {
        # 确定性计算:读取完整的计算结果,含事件级纳入情况与不确定性区间
        "name": "calculate",
        "description": "Read deterministic complete calculations, including event-level inclusions and uncertainty bounds.",
        "parameters": {
            "type": "object",
            "properties": {
                "view": {"type": "string", "enum": ["summary", "weeks", "events", "measures", "all"]},
                "date": {"type": "string"},
                "event_id": {"type": "string"},
            },
            "required": ["view"],
        },
    },
]


@dataclass
class InvestigationResult:
    """一次调查的结果。"""

    answer: str  # 最终回答文本
    citations: list[str]  # 通过校验的引用
    trace: list[dict[str, Any]]  # 完整调用轨迹(模型轮次与工具调用)
    audit: list[str]  # 引用审计中仍未解决的错误


class EvidenceTools:
    """把语料库、快照和计算结果封装成智能体可调用的工具集。"""

    def __init__(self, corpus: Corpus, snapshot: ReviewSnapshot, calculation: dict[str, Any]) -> None:
        self.corpus = corpus
        self.snapshot = snapshot
        self.calculation = calculation

    def invoke(self, name: str, args: dict[str, Any]) -> Any:
        """按名称分发工具调用;未知工具抛出 ValueError。"""
        if name == "search":
            # 结果条数限制在 1~30 之间,默认 12
            return self.corpus.search(str(args["query"]), max(1, min(int(args.get("limit", 12)), 30)))
        if name == "open_source":
            source_id = str(args["source_id"])
            first = max(1, int(args.get("first", 1)))
            # 单次最多读取 100 行,默认读 50 行
            last = min(first + 99, int(args.get("last", first + 49)))
            return self.corpus.open(source_id, first, last)
        if name == "related":
            event_id = str(args["event_id"])
            # 找不到事件会抛 StopIteration,由上层捕获并转成错误返回给模型
            event = next(item for item in self.snapshot.reconciliation.events if item.event_id == event_id)
            ids = set(event.supporting_mentions + event.opposing_mentions)
            mentions = [item for item in self.snapshot.extraction.events if item.mention_id in ids]
            return {
                "decision": event.model_dump(),
                "mentions": [
                    # 每条提及附带对应的原文摘录,便于模型核对
                    {**item.model_dump(), "source_excerpt": self.corpus.quote(item.anchor())}
                    for item in mentions
                ],
            }
        if name == "scan":
            kind = str(args["kind"])
            if kind == "sources":
                items = [
                    {"source_id": item.source_id, "filename": item.filename, "line_count": len(item.lines)}
                    for item in self.corpus.sources.values()
                ]
            elif kind == "events":
                items = [item.model_dump() for item in self.snapshot.reconciliation.events]
            else:
                items = [item.model_dump() for item in getattr(self.snapshot.extraction, kind)]
            # 按 date / service_type / theme 做子串过滤(不区分大小写);
            # 不同类别的日期字段名不同,这里做字段映射
            for key in ("date", "service_type", "theme"):
                if args.get(key):
                    field = "service_date" if key == "date" and kind == "events" else key
                    if key == "date" and kind == "observations":
                        field = "date"
                    if key == "date" and kind == "measures":
                        field = "completed_date"
                    items = [item for item in items if str(args[key]).lower() in str(item.get(field, "")).lower()]
            # 分页:offset 不小于 0,limit 限制在 1~100,默认 30
            offset = max(0, int(args.get("offset", 0)))
            limit = max(1, min(int(args.get("limit", 30)), 100))
            return {"total": len(items), "offset": offset, "items": items[offset : offset + limit], "next_offset": offset + limit if offset + limit < len(items) else None}
        if name == "calculate":
            view = str(args["view"])
            if view == "summary":
                # 摘要视图:去掉体量较大的明细(事件、周、目标)
                return {key: value for key, value in self.calculation.items() if key not in {"events", "weeks", "goals"}}
            if view == "weeks":
                return self.calculation["weeks"]
            if view == "events":
                # 可按日期或事件 ID 过滤
                return [
                    item for item in self.calculation["events"]
                    if (not args.get("date") or item["service_date"] == args["date"])
                    and (not args.get("event_id") or item["event_id"] == args["event_id"])
                ]
            if view == "measures":
                return self.calculation["measure_instances"]
            return self.calculation
        raise ValueError(f"Unknown tool: {name}")


def _citations(answer: str, corpus: Corpus) -> tuple[list[str], list[str]]:
    """从回答中提取形如 SOURCE_ID:L12 的引用并逐条校验,返回 (有效引用, 错误列表)。"""
    citations: list[str] = []
    errors: list[str] = []
    # 匹配 "来源ID:L12"、"来源ID:L12,L13"、"来源ID:L12-L15";前面不能紧跟标识符字符
    pattern = re.compile(r"(?<![A-Za-z0-9_.-])([A-Za-z0-9_.-]+):(L\d+(?:,L\d+|[-–]L\d+)*)")
    for match in pattern.finditer(answer):
        source_id, line_text = match.groups()
        numbers: list[int] = []
        for part in line_text.split(","):
            endpoints = [int(value) for value in re.findall(r"L(\d+)", part)]
            if len(endpoints) == 2:
                # 区间写法:不能倒置,跨度不得超过 100 行
                if endpoints[1] < endpoints[0] or endpoints[1] - endpoints[0] > 100:
                    errors.append(f"Invalid citation range: {source_id}:{part}")
                    continue
                numbers.extend(range(endpoints[0], endpoints[1] + 1))
            else:
                numbers.extend(endpoints)
        try:
            anchor = Anchor(source_id=source_id, lines=numbers)
            # 确认来源存在且行号没有越界
            corpus.validate_anchor(anchor)
            citations.append(anchor.reference())
        except ValueError as error:
            errors.append(str(error))
    if not citations:
        errors.append("No source citations in answer")
    return sorted(set(citations)), errors


def answer_question(
    question: str,
    model: ModelPort,
    tools: EvidenceTools,
    max_tool_calls: int = 12,
    max_model_turns: int = 6,
) -> InvestigationResult:
    """运行真正的"模型 + 工具"循环,时间与调用次数都有上限,且每次调用可审计。"""
    # 给模型的概览:先给出候选结论与索引,真正的事实仍需它自己去原文核实
    overview = {
        "period": tools.calculation["period"],
        "therapy_sessions": tools.calculation["therapy_sessions"],
        "sessions_by_type": tools.calculation["sessions_by_type"],
        "therapy_days": tools.calculation["therapy_days"],
        "therapy_minutes": tools.calculation["therapy_minutes"],
        "totals_complete": tools.calculation["totals_complete"],
        "unquantified_event_ids": tools.calculation["unquantified_event_ids"],
        "unresolved_mention_ids": tools.calculation["unresolved_mention_ids"],
        "coverage_gaps": tools.calculation["coverage_gaps"],
        # 周信息里去掉逐日明细以节省上下文
        "weeks": [
            {key: value for key, value in item.items() if key != "days"}
            for item in tools.calculation["weeks"]
        ],
        "event_index": [
            {key: item[key] for key in ("event_id", "service_date", "service_type", "minutes", "source_refs")}
            for item in tools.calculation["events"]
        ],
        "measure_index": [item.model_dump() for item in tools.snapshot.extraction.measures],
        "measure_instances": tools.calculation["measure_instances"],
        "findings": [item.model_dump() for item in tools.snapshot.findings],
    }
    history: list[dict[str, Any]] = [
        {"role": "user", "content": f"Question: {question}\n\nReview overview (candidate, inspect sources):\n{json.dumps(overview, ensure_ascii=False)}"}
    ]
    trace: list[dict[str, Any]] = []
    remaining = max_tool_calls  # 剩余的工具调用预算
    answer = ""
    for turn_number in range(1, max_model_turns + 1):
        # 预算用完后不再提供工具,迫使模型直接作答
        turn = model.generate(AGENT_SYSTEM, history, tools=TOOL_SPECS if remaining else None, max_tokens=6000)
        trace.append({
            "stage": "answer", "turn": turn_number, "usage": turn.usage,
            "stop_reason": turn.stop_reason, "model_text": turn.text,
            "tool_calls": [call.__dict__ for call in turn.tool_calls],
        })
        # 没有工具调用说明模型已给出最终回答
        if not turn.tool_calls:
            answer = turn.text
            break
        history.append(
            {"role": "assistant", "content": turn.text, "tool_calls": [call.__dict__ for call in turn.tool_calls], "response_items": turn.response_items}
        )
        for call in turn.tool_calls:
            if remaining <= 0:
                result = {"error": "Tool call budget exhausted; answer with current evidence."}
            else:
                remaining -= 1
                try:
                    result = tools.invoke(call.name, call.arguments)
                except (KeyError, ValueError, TypeError, StopIteration) as error:
                    # 工具出错不终止流程,把错误回传给模型让它自行调整
                    result = {"error": str(error)}
            serialized = json.dumps(result, ensure_ascii=False, default=str)
            history.append({"role": "tool", "tool_call_id": call.call_id, "content": serialized})
            trace.append({"stage": "tool", "name": call.name, "arguments": call.arguments, "result_chars": len(serialized), "result": result})
    # 轮次/预算耗尽仍没有回答:要求模型基于现有证据谨慎地给出最终答案
    if not answer:
        history.append({"role": "user", "content": "Tool or turn budget is exhausted. Give a cautious final answer using the evidence already available."})
        turn = model.generate(AGENT_SYSTEM, history, max_tokens=6000)
        answer = turn.text
        trace.append({"stage": "answer_final", "usage": turn.usage, "stop_reason": turn.stop_reason, "model_text": turn.text})
    # 引用审计:校验回答中的引用;有错误则让模型修正一次
    citations, errors = _citations(answer, tools.corpus)
    if errors:
        trace.append({"stage": "citation_audit", "errors": errors})
        history.append({"role": "assistant", "content": answer, "response_items": turn.response_items})
        history.append({"role": "user", "content": f"Citation audit failed: {errors}. Correct invalid/missing citations using only source IDs and line numbers already inspected. Return the full corrected answer."})
        revised = model.generate(AGENT_SYSTEM, history, max_tokens=6000)
        trace.append({"stage": "citation_repair", "usage": revised.usage, "stop_reason": revised.stop_reason, "model_text": revised.text})
        answer = revised.text
        # 修正后重新校验;若仍有错误会记录在 audit 中
        citations, errors = _citations(answer, tools.corpus)
    return InvestigationResult(answer=answer, citations=citations, trace=trace, audit=errors)
