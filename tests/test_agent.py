"""在不调用付费模型的前提下,验证智能体协议和来源引用校验门。"""

import unittest

from bb.agent import EvidenceTools, answer_question
from bb.model_provider import ModelTurn, ToolCall
from bb.models import Anchor, BatchExtraction, Reconciliation, ReviewSnapshot


class FakeCorpus:
    """假语料库:只含一个来源 SRC-1,只有第 1 行是合法锚点。"""

    sources = {"SRC-1": object()}

    def search(self, query: str, limit: int = 15):
        return [{"source_id": "SRC-1", "line": 1, "text": "A documented clinical observation."}]

    def validate_anchor(self, anchor: Anchor) -> None:
        if anchor.source_id != "SRC-1" or anchor.lines != [1]:
            raise ValueError("Invalid anchor")


class FakeModel:
    """假模型:第一轮发起一次 search 工具调用,第二轮给出带引用的最终回答。"""

    model_name = "offline-fake"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, system, history, tools=None, max_tokens=5000, json_mode=False):
        self.calls += 1
        if self.calls == 1:
            return ModelTurn(text="", tool_calls=[ToolCall("call-1", "search", {"query": "observation"})])
        return ModelTurn(text="The record contains a clinical observation [SRC-1:L1].")


class AgentTests(unittest.TestCase):
    def test_agent_selects_tool_and_validates_source_reference(self) -> None:
        """智能体应能选用工具,并对最终回答里的来源引用做校验。"""
        snapshot = ReviewSnapshot(
            source_hashes={}, extraction_model="offline-fake",
            extraction=BatchExtraction(), reconciliation=Reconciliation(),
        )
        # 构造一份最小的计算结果,满足 answer_question 读取概览所需的字段
        calculation = {
            "period": {}, "therapy_sessions": {}, "sessions_by_type": {},
            "therapy_days": {}, "therapy_minutes": {}, "weeks": [], "events": [],
            "measure_instances": [], "totals_complete": True,
            "unquantified_event_ids": [], "unresolved_mention_ids": [], "coverage_gaps": [],
        }
        model = FakeModel()
        result = answer_question("What is documented?", model, EvidenceTools(FakeCorpus(), snapshot, calculation))
        # 模型共被调用 2 次:一次发起工具调用,一次给出最终回答
        self.assertEqual(model.calls, 2)
        # 引用通过校验并被规范化
        self.assertEqual(result.citations, ["SRC-1:L1"])
        # 没有引用审计错误
        self.assertFalse(result.audit)
        # 轨迹中应包含工具调用阶段
        self.assertIn("tool", [item["stage"] for item in result.trace])


if __name__ == "__main__":
    unittest.main()
