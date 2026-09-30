"""命令行入口:运行一次可复现、来源可审计的审查。"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bb.agent import EvidenceTools, answer_question
from bb.compute import calculate_review
from bb.extract import extract_corpus
from bb.model_provider import make_model
from bb.models import ReviewSnapshot, validate_snapshot_reuse
from bb.reconcile import reconcile_events
from bb.report import model_call_count, render_answer_markdown
from bb.source import Corpus
from bb.time_audit import audit_time_scope


def _write_json(path: Path, value: Any) -> None:
    """把对象写成带缩进的 JSON 文件;"x" 模式保证不会覆盖已有文件。"""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=str)
        stream.write("\n")


def _read_questions(path: Path) -> list[dict[str, str]]:
    """读取问题文件。元素可以是字符串,或含 "question"(可选 "id")的对象。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Questions file must contain a JSON array")
    questions = []
    for index, item in enumerate(data):
        if isinstance(item, str):
            # 纯字符串问题:自动编号 Q-001、Q-002 ...
            questions.append({"id": f"Q-{index + 1:03d}", "question": item})
        elif isinstance(item, dict) and isinstance(item.get("question"), str):
            questions.append({"id": str(item.get("id", f"Q-{index + 1:03d}")), "question": item["question"]})
        else:
            raise ValueError(f"Invalid question at index {index}")
    return questions


def _run_directory(parent: Path) -> Path:
    """在输出根目录下创建本次运行专属目录:UTC 时间戳 + 6 位随机十六进制,避免冲突。"""
    parent.mkdir(parents=True, exist_ok=True)
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(3)
    target = parent / name
    target.mkdir(exist_ok=False)
    return target


def run(args: argparse.Namespace) -> Path:
    """执行一次完整审查:索引文档 → 抽取/复用快照 → 计算 → 逐题调查 → 写出结果。"""
    started = time.monotonic()
    output = _run_directory(args.output)

    def progress(message: str) -> None:
        """进度信息输出到 stderr,附 UTC 时间戳。"""
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)

    progress("Indexing source documents")
    # 为来源文档建立 SQLite 索引
    corpus = Corpus(args.documents, output / "index.sqlite3")
    model = make_model(args.provider, args.model, args.base_url)
    trace: list[dict[str, Any]] = []
    if args.snapshot:
        # 复用已有抽取快照:必须模型一致、所有来源哈希一致,否则抛错
        progress("Validating reusable abstraction")
        snapshot = ReviewSnapshot.model_validate_json(args.snapshot.read_text(encoding="utf-8"))
        validate_snapshot_reuse(snapshot, args.model, corpus.manifest())
    else:
        # 全新抽取流程:抽取 → 时间范围审计 → 事件对账
        cache_dir = args.output / "_stage_cache" if args.reuse_cache else None
        extraction, findings, extraction_trace = extract_corpus(
            corpus, model, max_chars=args.batch_chars,
            cache_dir=cache_dir, progress=progress,
        )
        trace.extend(extraction_trace)
        time_findings, time_trace = audit_time_scope(
            extraction, corpus, model, cache_dir=cache_dir, progress=progress,
        )
        findings.extend(time_findings)
        trace.extend(time_trace)
        reconciliation, more_findings, reconciliation_trace = reconcile_events(
            extraction, corpus, model, cache_dir=cache_dir, progress=progress,
        )
        trace.extend(reconciliation_trace)
        snapshot = ReviewSnapshot(
            source_hashes=corpus.manifest(),
            extraction_model=args.model,
            extraction=extraction,
            reconciliation=reconciliation,
            findings=findings + more_findings,
        )
    _write_json(output / "abstraction.json", snapshot.model_dump())
    progress("Calculating event and weekly totals")
    # 确定性计算事件与每周汇总(不让模型做加法)
    calculation = calculate_review(
        snapshot.reconciliation, snapshot.extraction, args.start, args.end, findings=snapshot.findings,
    )
    _write_json(output / "calculation.json", calculation)
    tools = EvidenceTools(corpus, snapshot, calculation)
    answers: list[dict[str, Any]] = []
    # --prepare-only 时只生成抽取结果,不回答问题
    questions = [] if args.prepare_only else _read_questions(args.questions)
    reports = output / "reports"
    if questions:
        reports.mkdir(exist_ok=False)
    offline_model_calls = model_call_count(trace)  # 离线阶段(抽取/对账)的模型调用数
    online_model_calls = 0  # 回答问题阶段的模型调用数
    for number, item in enumerate(questions, 1):
        progress(f"Investigating question {item['id']}")
        result = answer_question(
            item["question"], model, tools,
            max_tool_calls=args.max_tool_calls,
            max_model_turns=args.max_model_turns,
        )
        question_calls = model_call_count(result.trace)
        online_model_calls += question_calls
        answers.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answer": result.answer,
                "citations": result.citations,
                "citation_audit": result.audit,
                "online_model_calls": question_calls,
            }
        )
        # 每个问题单独生成一份 Markdown 报告
        markdown = render_answer_markdown(
            item["question"], result.answer, result.citations, corpus,
            question_calls, result.audit,
        )
        with (reports / f"question-{number:03d}.md").open("x", encoding="utf-8") as stream:
            stream.write(markdown)
        # 把该问题的轨迹并入总轨迹,并标注所属问题 ID
        trace.extend({**entry, "question_id": item["id"]} for entry in result.trace)
        progress(f"Completed question {item['id']}")
    _write_json(output / "answers.json", answers)
    # 轨迹以 JSONL 形式逐行保存
    with (output / "trace.jsonl").open("x", encoding="utf-8") as stream:
        for entry in trace:
            stream.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    # 汇总全程 token 用量
    usage = {
        "input_tokens": sum(entry.get("usage", {}).get("input_tokens", 0) for entry in trace),
        "output_tokens": sum(entry.get("usage", {}).get("output_tokens", 0) for entry in trace),
    }
    # 写出本次运行的元数据
    _write_json(
        output / "run.json",
        {
            "provider": args.provider,
            "model": args.model,
            "extraction_model": snapshot.extraction_model,
            "documents": len(corpus.sources),
            "questions": len(answers),
            "model_calls": online_model_calls,
            "offline_model_calls": offline_model_calls,
            "total_model_calls": offline_model_calls + online_model_calls,
            "usage": usage,
            "runtime_seconds": round(time.monotonic() - started, 2),
            "source_hashes": corpus.manifest(),
            "snapshot_reused": bool(args.snapshot),
            "cache_enabled": args.reuse_cache,
            "cost_usd": None,
            "cost_note": "Provider billing rate was not supplied; token usage is recorded for independent costing.",
        },
    )
    progress("Review run complete")
    return output


def main() -> None:
    """解析命令行参数并启动审查。"""
    parser = argparse.ArgumentParser(description="Auditable, source-grounded clinical record review")
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--questions", type=Path, help="JSON questions file, unless --prepare-only is used")
    parser.add_argument("--prepare-only", action="store_true", help="Build a fresh abstraction without asking questions yet")
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--provider", choices=("anthropic", "openai"), default="openai")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--snapshot", type=Path, help="Reuse an abstraction only if every source hash matches")
    parser.add_argument("--reuse-cache", action="store_true", help="Reuse prior model-stage results from this output root")
    parser.add_argument("--start", help="Inclusive review start date, YYYY-MM-DD")
    parser.add_argument("--end", help="Inclusive review end date, YYYY-MM-DD")
    parser.add_argument("--batch-chars", type=int, default=13500)
    parser.add_argument("--max-tool-calls", type=int, default=12)
    parser.add_argument("--max-model-turns", type=int, default=6)
    args = parser.parse_args()
    # --prepare-only 与 --questions 互斥;非 prepare-only 模式必须提供 --questions
    if args.prepare_only and args.questions:
        parser.error("--prepare-only and --questions cannot be used together")
    if not args.prepare_only and not args.questions:
        parser.error("--questions is required unless --prepare-only is used")
    target = run(args)
    print(target)


if __name__ == "__main__":
    main()
