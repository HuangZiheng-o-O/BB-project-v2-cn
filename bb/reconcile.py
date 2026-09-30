"""把各来源的提及对账为可审查的、字段级的事件结论。"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from pydantic import ValidationError

from bb.cache import StageCache
from bb.model_provider import ModelPort
from bb.models import AuditFinding, BatchExtraction, EventMention, Reconciliation, ResolvedEvent
from bb.repair import generate_checked_json
from bb.source import Corpus


# 对账阶段的系统提示词(发给模型的原文,保持英文不变)
RECONCILIATION_SYSTEM = """You reconcile claims about clinical encounters, not documents. Return one JSON object:
{"events":[{"event_id":"...","service_date":"YYYY-MM-DD or null","service_type":"...","disposition":"delivered|not_delivered|uncertain","patient_therapy":"yes|no|uncertain","interval_options":[[{"start":"HH:MM","end":"HH:MM"}]],"excluded_intervals":[{"start":"HH:MM","end":"HH:MM"}],"supporting_mentions":["id"],"opposing_mentions":["id"],"decision_notes":"..."}],"unresolved_mention_ids":[]}.

Produce exactly one event for EVERY supplied group_id, using that group_id as event_id. Include non-therapy encounters and no-shows as events too. Put EVERY supplied mention ID in supporting_mentions or opposing_mentions, exactly once, and explain important inclusion/exclusion decisions. These lists mean considered evidence for or against the chosen event decision, not a document credibility ranking. A clinical note, attendance row, billing record, draft, and later correction can refer to the same encounter, not separate visits. A source received later does not automatically prevail.

For delivered patient psychotherapy, interval_options contains one or more POSSIBLE sets of patient-present time spans. One unambiguous event has one option. If two credible patient-contact records conflict without an explicit amendment, preserve each plausible option rather than choosing or adding them. If no patient therapy occurred, interval_options must be empty.

Group breaks, network disconnects, and partner-only segments cannot count as patient therapy. Put shared breaks/disconnections in excluded_intervals if they overlap a broad patient interval. If actual_intervals already exclude a gap, retaining the gap in excluded_intervals is harmless. For a mixed family encounter, use only the patient's portion.

An explicit correction supersedes ONLY the named field for that earlier event. A later retransmission of the original record does not undo the correction and does not create a new event. A schedule, charge, authorization, unsigned template, or administrative contact alone does not establish that patient therapy was delivered. Signed but conflicting clinical records may leave duration unresolved.

Do not calculate minutes or weekly totals. If evidence cannot establish delivery or patient presence, set uncertain and state why. Preserve concrete, source-grounded distinctions; do not silently resolve a conflict with a universal document priority rule. Return JSON only."""

# 缓存版本号:修改对账规则/提示词语义时需要更新,使旧缓存失效
RECONCILIATION_CACHE_VERSION = "clinical-interval-conflict-v2"


def group_mentions(extraction: BatchExtraction) -> dict[str, list[EventMention]]:
    """把事件提及按"患者:就诊标识"分组,同一组的提及指向同一次诊疗事件。"""
    groups: dict[str, list[EventMention]] = defaultdict(list)
    known_patients = {item.patient_id for item in extraction.events if item.patient_id}
    # 语料里只有一位患者时,缺失 patient_id 的提及默认归属于这位患者
    sole_patient = next(iter(known_patients)) if len(known_patients) == 1 else None
    # 建立 (患者, 预约号) → 就诊号 的映射,让只有预约号的记录能关联到对应就诊
    appointment_to_encounter: dict[tuple[str, str], set[str]] = defaultdict(set)
    for mention in extraction.events:
        patient = mention.patient_id or sole_patient
        if patient and mention.appointment_id and mention.encounter_id:
            appointment_to_encounter[(patient, mention.appointment_id)].add(mention.encounter_id)
    for mention in extraction.events:
        patient = mention.patient_id or sole_patient or f"unknown:{mention.source_id}"
        if mention.encounter_id:
            identity = mention.encounter_id
        elif mention.appointment_id:
            linked = appointment_to_encounter.get((patient, mention.appointment_id), set())
            # 只有预约号恰好对应唯一就诊号时才合并;否则用预约号自身作为标识
            identity = next(iter(linked)) if len(linked) == 1 else mention.appointment_id
        else:
            # 缺少标识的提及单独成组,不会仅凭日期就合并。
            identity = f"unlinked:{mention.mention_id}"
        key = f"{patient}:{identity}"
        groups[key].append(mention)
    return dict(groups)


def _group_payload(group_id: str, mentions: list[EventMention], corpus: Corpus) -> dict[str, Any]:
    """把一组提及整理成发给模型的载荷,并附上原文摘录。"""
    payload = []
    for mention in mentions:
        data = mention.model_dump(exclude={"note"}, exclude_none=True)
        # 原文摘录最多 1200 字符,备注最多 500 字符,控制提示词长度
        data["source_excerpt"] = corpus.quote(mention.anchor())[:1200]
        if mention.note:
            data["note"] = mention.note[:500]
        payload.append(data)
    return {"group_id": group_id, "mentions": payload}


def _batches(groups: dict[str, list[EventMention]], corpus: Corpus, max_chars: int = 18000) -> list[list[dict]]:
    """把各组载荷按字符数上限切成若干批,每批交给模型一次对账。"""
    output: list[list[dict]] = []
    pending: list[dict] = []
    length = 0
    for group_id, mentions in groups.items():
        item = _group_payload(group_id, mentions, corpus)
        item_length = len(json.dumps(item, ensure_ascii=False))
        # 当前批再加入就会超限 → 先收尾当前批,开启新批
        if pending and length + item_length > max_chars:
            output.append(pending)
            pending, length = [], 0
        pending.append(item)
        length += item_length
    if pending:
        output.append(pending)
    return output


def reconcile_events(
    extraction: BatchExtraction,
    corpus: Corpus,
    model: ModelPort,
    cache_dir: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[Reconciliation, list[AuditFinding], list[dict[str, Any]]]:
    """调用模型对每组提及做对账,逐条校验后汇总;返回 (对账结果, 审计发现, 调用轨迹)。"""
    groups = group_mentions(extraction)
    findings: list[AuditFinding] = []
    trace: list[dict[str, Any]] = []
    resolved: dict[str, ResolvedEvent] = {}
    batches = _batches(groups, corpus)
    cache = StageCache(cache_dir) if cache_dir else None
    for number, batch in enumerate(batches, 1):
        if progress:
            progress(f"Reconciling batch {number}/{len(batches)}")
        expected = {item["group_id"] for item in batch}  # 本批必须全部给出结论的组 ID
        prompt = f"Reconcile group batch {number}/{len(batches)}:\n{json.dumps(batch, ensure_ascii=False)}"
        cache_path = cache.path(
            "reconcile", model.model_name, RECONCILIATION_SYSTEM + RECONCILIATION_CACHE_VERSION, prompt,
        ) if cache else None
        def validated_events(data: dict[str, Any]) -> tuple[dict[str, ResolvedEvent], list[str]]:
            """校验模型返回的对账结果,返回 (通过校验的事件, 错误列表)。"""
            accepted: dict[str, ResolvedEvent] = {}
            errors: list[str] = []
            rows = data.get("events")
            if not isinstance(rows, list):
                return {}, ["events must be an array"]
            for index, raw in enumerate(rows):
                try:
                    event = ResolvedEvent.model_validate(raw)
                    # 事件 ID 必须是本批提供的组 ID
                    if event.event_id not in expected:
                        raise ValueError(f"Unknown group ID {event.event_id}")
                    member_ids = {mention.mention_id for mention in groups[event.event_id]}
                    listed_ids = event.supporting_mentions + event.opposing_mentions
                    # 组内每条提及必须在支持/反对列表中恰好出现一次
                    if set(listed_ids) != member_ids or len(listed_ids) != len(member_ids):
                        raise ValueError("Decision must classify every mention exactly once")
                    # 逻辑矛盾:未提供的事件不能同时确认患者接受了治疗
                    if event.disposition == "not_delivered" and event.patient_therapy == "yes":
                        raise ValueError("Non-delivered event cannot be confirmed patient therapy")
                    members = groups[event.event_id]
                    # 收集各份"临床记录、患者在场、有实际时间段"的时间区间(去重)
                    clinical_intervals = {
                        tuple((span.start, span.end) for span in mention.actual_intervals)
                        for mention in members
                        if mention.document_role == "clinical"
                        and mention.patient_present is True
                        and mention.actual_intervals
                    }
                    explicit_correction = any(mention.correction_field for mention in members)
                    # 多份临床记录的时间冲突且没有明确更正时,必须把每种时间都保留为备选方案,
                    # 不能由模型擅自选一个或相加
                    if len(clinical_intervals) > 1 and not explicit_correction:
                        chosen_intervals = {
                            tuple((span.start, span.end) for span in option)
                            for option in event.interval_options
                        }
                        if not clinical_intervals.issubset(chosen_intervals):
                            raise ValueError(
                                "Conflicting clinical patient-contact intervals require separate interval_options"
                            )
                    if event.event_id in accepted:
                        raise ValueError("Duplicate decision for one event group")
                    accepted[event.event_id] = event
                except (ValidationError, ValueError, KeyError) as error:
                    errors.append(f"event {index}: {error}")
            # 每个组都必须有结论
            missing = expected - set(accepted)
            if missing:
                errors.append(f"Missing group IDs: {sorted(missing)}")
            return accepted, errors

        cached = cache.read(cache_path) if cache_path else None
        # 缓存内容若不能通过当前校验规则,则视为未命中
        if cached is not None and validated_events(cached)[1]:
            cached = None
        if cached is None:
            data, calls = generate_checked_json(
                model, RECONCILIATION_SYSTEM, prompt,
                lambda value: validated_events(value)[1], max_tokens=10000,
            )
        else:
            data, calls = cached, [{"cache_hit": True}]
        trace.extend({**call, "stage": "reconcile", "batch": number} for call in calls)
        accepted, _ = validated_events(data)
        # 校验通过后才写缓存
        if cache_path and not cache_path.exists():
            StageCache.write(cache_path, data)
        for event in accepted.values():
            # 声称患者接受了治疗却没有带来源的时间区间 → 记为"无法量化"的审计发现
            if event.patient_therapy == "yes" and not event.interval_options:
                findings.append(
                    AuditFinding(
                        code="unquantified_event",
                        detail=f"Patient therapy is claimed without a sourced interval for {event.event_id}",
                        source_refs=[mention.anchor().reference() for mention in groups[event.event_id]],
                    )
                )
        resolved.update(accepted)
        if progress:
            progress(f"Completed reconciliation batch {number}/{len(batches)}")
    # 没有得到结论的组,其所有提及计入"未解决"
    unresolved = [
        mention.mention_id
        for group_id, mentions in groups.items()
        if group_id not in resolved
        for mention in mentions
    ]
    return Reconciliation(events=list(resolved.values()), unresolved_mention_ids=unresolved), findings, trace
