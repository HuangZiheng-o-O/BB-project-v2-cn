"""核实抽取出的时钟区间描述的是患者的实际接触时间,还是服务/活动的时间范围。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from bb.cache import StageCache
from bb.model_provider import ModelPort
from bb.models import AuditFinding, BatchExtraction
from bb.repair import generate_checked_json
from bb.source import Corpus


# 时间范围审计的系统提示词(发给模型的原文,保持英文不变)
TIME_SCOPE_SYSTEM = """Audit the provenance and scope of extracted clock intervals. Return one JSON object:
{"decisions":[{"mention_id":"...","patient_actual_supported":true|false|null,"reason":"..."}]}.

For EVERY supplied mention, determine whether its actual_intervals are explicitly supported as THIS PATIENT's actual arrival/departure or direct therapeutic contact in the original source. A group schedule, session opening/closing time, facilitator activity window, or session break does not by itself establish the patient's actual presence for that full period. Participation narrative without patient-specific clock times also does not establish full attendance. Conversely, explicit patient contact times do support actual_intervals. Read the full numbered source and distinguish its time fields by role. Use false when the clock spans describe only the service/group rather than the patient's own time; use null if genuinely unclear. Do not calculate therapy minutes or adjudicate other sources. Return JSON only."""


def audit_time_scope(
    extraction: BatchExtraction,
    corpus: Corpus,
    model: ModelPort,
    cache_dir: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[AuditFinding], list[dict[str, Any]]]:
    """审计小组治疗记录里的"实际时间段"是否真的属于该患者;返回 (审计发现, 调用轨迹)。

    注意:判定为"不属于患者"的提及,其 actual_intervals 会被就地清空。
    """
    # 只审计这类提及:临床记录、小组治疗,且同时有"实际时间段"和"计划时间段"
    # (两者并存时,最容易把小组整体时间误当成患者自己的时间)
    candidates = [
        mention for mention in extraction.events
        if mention.document_role == "clinical"
        and "group" in (mention.service_type or "").lower()
        and mention.actual_intervals
        and mention.scheduled_intervals
    ]
    if not candidates:
        return [], []
    if progress:
        progress(f"Auditing patient-time scope for {len(candidates)} clinical group mentions")
    # 给模型的载荷:已抽取的时间段 + 文档全文,让它对照原文判断
    payload = [
        {
            "mention_id": mention.mention_id,
            "source_id": mention.source_id,
            "extracted_actual_intervals": [item.model_dump() for item in mention.actual_intervals],
            "scheduled_intervals": [item.model_dump() for item in mention.scheduled_intervals],
            "source": corpus.get(mention.source_id).formatted(),
        }
        for mention in candidates
    ]
    serialized = json.dumps(payload, ensure_ascii=False)
    cache = StageCache(cache_dir) if cache_dir else None
    cache_path = cache.path("time_scope", model.model_name, TIME_SCOPE_SYSTEM, serialized) if cache else None
    result = cache.read(cache_path) if cache_path else None

    expected = {mention.mention_id for mention in candidates}  # 必须逐一给出判定的提及 ID

    def validation_errors(data: dict[str, Any]) -> list[str]:
        """校验模型输出:结构正确、判定值合法、ID 不重复且与候选集完全一致。"""
        decisions = data.get("decisions")
        if not isinstance(decisions, list):
            return ["decisions must be an array"]
        errors: list[str] = []
        identifiers: list[str] = []
        for index, decision in enumerate(decisions):
            if not isinstance(decision, dict):
                errors.append(f"decisions[{index}] must be an object")
                continue
            identifiers.append(str(decision.get("mention_id")))
            verdict = decision.get("patient_actual_supported")
            # 判定值只能是 true / false / null
            if verdict is not True and verdict is not False and verdict is not None:
                errors.append(f"decisions[{index}].patient_actual_supported must be true, false, or null")
        if len(identifiers) != len(set(identifiers)):
            errors.append("Duplicate mention IDs in time-scope decisions")
        # 不能遗漏,也不能出现未知 ID
        if set(identifiers) != expected:
            errors.append(f"Missing IDs: {sorted(expected - set(identifiers))}; unknown IDs: {sorted(set(identifiers) - expected)}")
        return errors

    # 缓存内容若不能通过当前校验,则视为未命中
    if result is not None and validation_errors(result):
        result = None
    if result is None:
        result, calls = generate_checked_json(
            model, TIME_SCOPE_SYSTEM,
            f"Audit these extracted time claims:\n{serialized}",
            validation_errors,
            max_tokens=2000,
        )
    else:
        calls = [{"cache_hit": True}]
    trace = [{**call, "stage": "time_scope"} for call in calls]
    decisions = result.get("decisions", [])
    by_id = {item.get("mention_id"): item for item in decisions if isinstance(item, dict)}
    findings: list[AuditFinding] = []
    for mention in candidates:
        decision = by_id[mention.mention_id]
        verdict = decision.get("patient_actual_supported")
        if verdict is False:
            # 判定为"只是服务/小组层面的时间":清空患者实际时间段,并记录原值以便审计
            original = [item.model_dump() for item in mention.actual_intervals]
            mention.actual_intervals = []
            findings.append(
                AuditFinding(
                    code="time_scope_reclassified",
                    detail=f"{mention.mention_id}: candidate patient intervals {original} were service-level time; {decision.get('reason', '')}",
                    source_refs=[mention.anchor().reference()],
                )
            )
        elif verdict is None:
            # 无法判断:保留原值,但记为"时间范围不确定"(会使汇总被标记为不完整)
            findings.append(
                AuditFinding(
                    code="time_scope_uncertain",
                    detail=f"{mention.mention_id}: patient-specific basis for extracted time remains unclear; {decision.get('reason', '')}",
                    source_refs=[mention.anchor().reference()],
                )
            )
    # 结果校验通过后才写缓存
    if cache_path and not cache_path.exists():
        StageCache.write(cache_path, result)
    trace.append({"stage": "time_scope_decisions", "decisions": decisions})
    return findings, trace
