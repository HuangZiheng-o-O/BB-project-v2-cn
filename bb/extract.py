"""带来源锚点的候选信息抽取;不含任何针对特定答案的解析规则。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, ValidationError

from bb.cache import StageCache
from bb.model_provider import ModelPort
from bb.models import AuditFinding, BatchExtraction, EventMention, MeasureMention, Observation, PlanGoal
from bb.repair import generate_checked_json
from bb.source import Corpus


# 抽取阶段的系统提示词(发给模型的原文,保持英文不变)
EXTRACTION_SYSTEM = """You are a clinical records evidence extractor. Produce source claims, never a final patient answer.

Return one JSON object with exactly four arrays: events, goals, measures, observations. No markdown.
Every item MUST cite the provided DOCUMENT source_id and 1-based line numbers. The lines field is an array of integers, for example [3,4,5], not strings such as ["L0003-L0005"]. Cite concise decisive lines (at most 12 per item). Do not invent a source or a line.

events: one mention per specific patient encounter or appointment described by each source. A multi-row register yields one mention per row. Fields: source_id, lines, patient_id (stable chart/MRN identifier if stated), encounter_id, appointment_id, service_date (YYYY-MM-DD), service_type (individual/group/family/medication/collateral/care_coordination/other), document_role (clinical/attendance/schedule/correction/charge/draft/administrative), status (delivered/attended/no_show/cancelled/scheduled/posted/draft/correction/unknown), patient_present (true/false/null), actual_intervals, scheduled_intervals, nontherapy_intervals, correction_field, correction_value, duplicate_of, note. Each interval is {start:"HH:MM",end:"HH:MM"}. Empty arrays and nulls are allowed.
actual_intervals represent documented patient-present clinical contact only. If a record reports only scheduled time, put it in scheduled_intervals. For a mixed family session, actual_intervals include only the patient's portion. A connection interruption or group break belongs in nontherapy_intervals; separate connection segments in one appointment remain one mention. A correction is a statement about an earlier field, not another service. A charge, unsigned template, authorization, or administrative call does not prove delivered patient therapy.

goals: signed treatment-plan participation goals only. Fields: source_id, lines, effective_from, effective_to, period, minimum_days, minimum_minutes, included_services, excluded_services, description. Set period to week_monday_sunday only when the plan explicitly defines a Monday-Sunday week, weekly when it says weekly without a specified week boundary, or null when absent. Leave unknown fields null. Do not infer targets from an authorization quantity.

measures: one mention per actual questionnaire or explicitly identified copy. Fields: source_id, lines, instrument, form_id, completed_date, score, copied_from_form, note. The receipt/import date is not a new completion date.

observations: concise patient-specific symptom, functional course, safety, treatment-change, or reason-for-extra-contact claims. Fields: source_id, lines, date, subject, theme, statement, polarity (positive/negative/uncertain/planned). Preserve who reported the observation and whether it is a plan rather than an achieved result in statement. Do not convert collateral observations into direct patient reports. Prefer a few important observations per source over generic repetition.

Extract all concrete encounters/appointment rows even when they are not therapy. Keep conflicting claims from different sources. Do not decide which source wins and do not calculate weekly totals."""

# 补漏审计阶段的系统提示词:在抽取提示词基础上追加说明
REPAIR_SYSTEM = EXTRACTION_SYSTEM + "\n\nThis pass audits one source that explicitly names an encounter or appointment but yielded no event mention in a larger batch. Extract every concrete event claim in this source. An accompanying clinician note for an existing encounter is still an event mention, even when it is not an additional visit."


def _explicit_contact_ids(text: str) -> set[str]:
    """找出文本中明确标注的就诊/预约编号,用于召回审计(检查有没有漏抽)。"""
    return {
        match.group(1)
        for match in re.finditer(
            # 形如 "Encounter ID: BH-E123"、"appointment #A-01-2" 的标注
            r"\b(?:encounter|appointment)(?:\s+(?:id|number))?\s*[:#]?\s*([A-Z][A-Z0-9]*-[A-Z0-9-]+)\b",
            text,
            flags=re.IGNORECASE,
        )
    }


T = TypeVar("T", bound=BaseModel)


def _validated_items(
    data: dict[str, Any],
    key: str,
    model_type: type[T],
    corpus: Corpus,
    allowed_sources: set[str],
    findings: list[AuditFinding],
) -> list[T]:
    """校验模型返回的某一类条目(events/goals/...),丢弃无效条目并把原因记入 findings。"""
    values = data.get(key, [])
    if not isinstance(values, list):
        findings.append(AuditFinding(code="invalid_extraction_array", detail=f"{key} is not a list"))
        return []
    valid: list[T] = []
    for index, raw in enumerate(values):
        try:
            item = model_type.model_validate(raw)
            anchor = item.anchor()
            # 条目引用的来源必须属于本批次,防止模型引用没给它看的文档
            if anchor.source_id not in allowed_sources:
                raise ValueError(f"{anchor.source_id} was not in this extraction batch")
            # 来源必须存在且行号不越界
            corpus.validate_anchor(anchor)
            if isinstance(item, EventMention):
                # 事件提及生成稳定 ID
                item.finalize_id()
            valid.append(item)
        except (ValueError, ValidationError, KeyError) as error:
            findings.append(
                AuditFinding(
                    code="invalid_source_candidate",
                    detail=f"{key}[{index}] rejected: {error}",
                )
            )
    return valid


def _payload_errors(
    data: dict[str, Any], corpus: Corpus, allowed_sources: set[str], required_ids: set[str] | None = None,
) -> list[str]:
    """检查一次抽取输出是否合格,返回错误列表(空列表表示合格)。"""
    # 四个数组必须齐全
    errors = [f"Missing required array: {key}" for key in ("events", "goals", "measures", "observations") if key not in data]
    findings: list[AuditFinding] = []
    events = _validated_items(data, "events", EventMention, corpus, allowed_sources, findings)
    _validated_items(data, "goals", PlanGoal, corpus, allowed_sources, findings)
    _validated_items(data, "measures", MeasureMention, corpus, allowed_sources, findings)
    _validated_items(data, "observations", Observation, corpus, allowed_sources, findings)
    # 任何被拒绝的条目都算错误,促使模型修复
    errors.extend(item.detail for item in findings)
    if required_ids:
        # 补漏阶段:要求必须覆盖指定的就诊/预约编号
        represented = {
            identity for item in events
            for identity in (item.encounter_id, item.appointment_id) if identity
        }
        errors.extend(f"Missing labeled encounter or appointment: {identity}" for identity in sorted(required_ids - represented))
    return errors


def extract_corpus(
    corpus: Corpus,
    model: ModelPort,
    max_chars: int = 13500,
    cache_dir: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[BatchExtraction, list[AuditFinding], list[dict[str, Any]]]:
    """对整个语料做分批抽取,并对明确标注编号却未被抽到的来源做补漏;返回 (抽取结果, 审计发现, 轨迹)。"""
    all_events: list[EventMention] = []
    all_goals: list[PlanGoal] = []
    all_measures: list[MeasureMention] = []
    all_observations: list[Observation] = []
    findings: list[AuditFinding] = []
    trace: list[dict[str, Any]] = []
    batches = corpus.extraction_batches(max_chars=max_chars)
    cache = StageCache(cache_dir) if cache_dir else None
    # 第一步:按批次抽取
    for batch_number, batch in enumerate(batches, 1):
        if progress:
            progress(f"Extracting batch {batch_number}/{len(batches)}")
        # 本批次里出现的文档 ID(模型只允许引用这些)
        allowed = set(re.findall(r"^DOCUMENT (\S+)", batch, re.MULTILINE))
        cache_path = cache.path("extract", model.model_name, EXTRACTION_SYSTEM, batch) if cache else None
        result = cache.read(cache_path) if cache_path else None
        # 缓存内容若不能通过当前校验,则视为未命中
        if result is not None and _payload_errors(result, corpus, allowed):
            result = None
        if result is None:
            result, calls = generate_checked_json(
                model,
                EXTRACTION_SYSTEM,
                f"Extract evidence from source batch {batch_number}/{len(batches)}:\n\n{batch}",
                lambda data: _payload_errors(data, corpus, allowed),
                max_tokens=10000,
            )
        else:
            calls = [{"cache_hit": True}]
        trace.extend({**call, "stage": "extract", "batch": batch_number} for call in calls)
        prior_findings = len(findings)
        all_events.extend(_validated_items(result, "events", EventMention, corpus, allowed, findings))
        all_goals.extend(_validated_items(result, "goals", PlanGoal, corpus, allowed, findings))
        all_measures.extend(_validated_items(result, "measures", MeasureMention, corpus, allowed, findings))
        all_observations.extend(
            _validated_items(result, "observations", Observation, corpus, allowed, findings)
        )
        # 只有本批没有新增审计发现(输出完全干净)时才写缓存
        if cache_path and not cache_path.exists() and len(findings) == prior_findings:
            StageCache.write(cache_path, result)
        if progress:
            progress(f"Completed extraction batch {batch_number}/{len(batches)}")
    # 第二步:召回审计——对文中明确写了编号、却没抽到对应事件的来源单独重抽
    for source in corpus.sources.values():
        explicit_ids = _explicit_contact_ids("\n".join(source.lines))
        extracted_ids = {
            identity
            for item in all_events if item.source_id == source.source_id
            for identity in (item.encounter_id, item.appointment_id) if identity
        }
        missing_ids = explicit_ids - extracted_ids
        if not missing_ids:
            continue
        if progress:
            progress(f"Checking event coverage for source {source.source_id}: {len(missing_ids)} unrepresented IDs")
        payload = source.formatted()
        cache_path = cache.path("extract_repair", model.model_name, REPAIR_SYSTEM, payload) if cache else None
        result = cache.read(cache_path) if cache_path else None
        if result is not None and _payload_errors(result, corpus, {source.source_id}, missing_ids):
            result = None
        if result is None:
            result, calls = generate_checked_json(
                model, REPAIR_SYSTEM,
                f"Audit this source for all concrete event mentions:\n\n{payload}",
                lambda data: _payload_errors(data, corpus, {source.source_id}, missing_ids),
                max_tokens=5000,
            )
        else:
            calls = [{"cache_hit": True}]
        trace.extend({**call, "stage": "extract_repair", "source_id": source.source_id} for call in calls)
        # 只采纳那些涉及缺失编号的补抽事件,避免与第一步重复
        repaired = [
            item for item in _validated_items(result, "events", EventMention, corpus, {source.source_id}, findings)
            if {item.encounter_id, item.appointment_id} & missing_ids
        ]
        all_events.extend(repaired)
        recovered_ids = {
            identity for item in repaired
            for identity in (item.encounter_id, item.appointment_id) if identity
        }
        still_missing = missing_ids - recovered_ids
        # 补漏后仍有缺失编号:宁可报错终止,也不带着遗漏继续
        if still_missing:
            raise ValueError(f"Validated source repair omitted labeled IDs: {sorted(still_missing)}")
        if cache_path and not cache_path.exists():
            StageCache.write(cache_path, result)
    # 按 mention_id 去重(同一条提及可能被多个批次/补漏重复抽到)
    unique_events = {item.mention_id: item for item in all_events}
    extraction = BatchExtraction(
        events=list(unique_events.values()),
        goals=all_goals,
        measures=all_measures,
        observations=all_observations,
    )
    return extraction, findings, trace
