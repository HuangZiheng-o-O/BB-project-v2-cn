"""本地小型 Gradio 页面,每次提一个问题。"""

from __future__ import annotations

import argparse
import json
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bb.agent import EvidenceTools, answer_question
from bb.compute import calculate_review
from bb.model_provider import ModelPort, make_model
from bb.models import ReviewSnapshot, validate_snapshot_reuse
from bb.report import model_call_count, render_answer_markdown
from bb.source import Corpus


def _new_directory(parent: Path) -> Path:
    """在父目录下创建以 UTC 时间戳 + 8 位随机十六进制命名的新目录。"""
    parent.mkdir(parents=True, exist_ok=True)
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(4)
    target = parent / name
    target.mkdir(exist_ok=False)
    return target


class ReviewSession:
    """复用一份已校验的抽取快照、计算结果、来源索引和模型客户端。"""

    def __init__(
        self,
        documents: Path,
        run_path: Path,
        output_root: Path,
        model: ModelPort,
        max_tool_calls: int = 12,
        max_model_turns: int = 6,
    ) -> None:
        run_metadata = json.loads((run_path / "run.json").read_text(encoding="utf-8"))
        # 所选运行结果必须由同一模型产生,否则不能混用
        if run_metadata["model"] != model.model_name:
            raise ValueError("The selected run and answer model differ; use a run produced by this model")
        self.directory = _new_directory(output_root)
        self.corpus = Corpus(documents, self.directory / "index.sqlite3")
        self.snapshot = ReviewSnapshot.model_validate_json(
            (run_path / "abstraction.json").read_text(encoding="utf-8")
        )
        # 校验:模型一致、来源哈希一致、快照中没有被拒绝的阶段输出
        validate_snapshot_reuse(self.snapshot, model.model_name, self.corpus.manifest())
        recorded = json.loads((run_path / "calculation.json").read_text(encoding="utf-8"))
        period = recorded["period"]
        # 用当前代码重新计算一遍,必须与记录一致,防止代码或快照被改动后结果悄悄漂移
        recalculated = calculate_review(
            self.snapshot.reconciliation,
            self.snapshot.extraction,
            period["start"],
            period["end"],
            findings=self.snapshot.findings,
        )
        if recalculated != recorded:
            raise ValueError("Calculation does not match the snapshot and current code")
        self.tools = EvidenceTools(self.corpus, self.snapshot, recalculated)
        self.model = model
        self.max_tool_calls = max_tool_calls
        self.max_model_turns = max_model_turns

    def ask(self, question: str) -> tuple[str, str]:
        """回答一个问题,返回 (Markdown 报告文本, 报告文件路径)。"""
        clean_question = (question or "").strip()
        if not clean_question:
            raise ValueError("Enter a question before submitting")
        started = time.monotonic()
        result = answer_question(
            clean_question,
            self.model,
            self.tools,
            max_tool_calls=self.max_tool_calls,
            max_model_turns=self.max_model_turns,
        )
        online_model_calls = model_call_count(result.trace)
        markdown = render_answer_markdown(
            clean_question, result.answer, result.citations, self.corpus,
            online_model_calls, result.audit,
        )

        # 每次提问在 answers 目录下单独存档:报告、轨迹、运行元数据
        answer_dir = _new_directory(self.directory / "answers")
        markdown_path = answer_dir / "answer.md"
        with markdown_path.open("x", encoding="utf-8") as stream:
            stream.write(markdown)
        with (answer_dir / "trace.jsonl").open("x", encoding="utf-8") as stream:
            for item in result.trace:
                stream.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
        usage: dict[str, int] = {
            "input_tokens": sum(item.get("usage", {}).get("input_tokens", 0) for item in result.trace),
            "output_tokens": sum(item.get("usage", {}).get("output_tokens", 0) for item in result.trace),
        }
        metadata: dict[str, Any] = {
            "question": clean_question,
            "model": self.model.model_name,
            "citations": result.citations,
            "citation_audit": result.audit,
            "online_model_calls": online_model_calls,
            "usage": usage,
            "runtime_seconds": round(time.monotonic() - started, 2),
            "source_hashes": self.snapshot.source_hashes,
        }
        with (answer_dir / "run.json").open("x", encoding="utf-8") as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        return markdown, str(markdown_path)


def build_app(session: ReviewSession):
    """构建 Gradio 界面:一个问题输入框、一个提交按钮、回答展示区和 Markdown 下载。"""
    # 延迟导入:gradio 是可选依赖,只有启动网页时才需要
    import gradio as gr

    with gr.Blocks(title="Clinical Evidence Review") as app:
        gr.Markdown("# Clinical Evidence Review\nAsk a question about the loaded records.")
        question = gr.Textbox(label="Question", lines=3, placeholder="What does the record establish?")
        submit = gr.Button("Ask", variant="primary")
        answer = gr.Markdown(label="Answer")
        download = gr.File(label="Download Markdown", interactive=False)
        # concurrency_limit=1:同一时间只处理一个问题
        submit.click(
            fn=session.ask,
            inputs=question,
            outputs=[answer, download],
            concurrency_limit=1,
        )
    return app


def main() -> None:
    """解析参数,加载会话,并在本机 127.0.0.1 上启动网页(不对外分享)。"""
    parser = argparse.ArgumentParser(description="Local question-answer page for a saved clinical review")
    parser.add_argument("--documents", type=Path, default=Path("data"))
    parser.add_argument("--run", type=Path, required=True, help="Directory from a fresh bb-review run with the same model")
    parser.add_argument("--output", type=Path, default=Path("runs/web"))
    parser.add_argument("--provider", choices=("anthropic", "openai"), default="openai")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    model = make_model(args.provider, args.model, args.base_url)
    session = ReviewSession(args.documents, args.run, args.output, model)
    build_app(session).launch(server_name="127.0.0.1", server_port=args.port, share=False, show_error=True)


if __name__ == "__main__":
    main()
