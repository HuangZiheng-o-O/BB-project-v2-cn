"""对已对账、可溯源的诊疗事件结论做确定性的算术汇总。"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, time, timedelta
import re
from typing import Any

from bb.models import AuditFinding, BatchExtraction, Reconciliation, ResolvedEvent, TimeSpan


def _instant(day: date, clock: str) -> datetime:
    """把某一天和 HH:MM 时钟字符串组合成具体时刻。"""
    return datetime.combine(day, time.fromisoformat(clock))


def _ranges(day: date, spans: list[TimeSpan]) -> list[tuple[datetime, datetime]]:
    """把一组 TimeSpan 转成具体的 (开始, 结束) 时刻区间。"""
    ranges = []
    for span in spans:
        start, end = _instant(day, span.start), _instant(day, span.end)
        # 结束不晚于开始,视为跨午夜,结束顺延到次日
        if end <= start:
            end += timedelta(days=1)
        ranges.append((start, end))
    return ranges


def _union(ranges: list[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime]]:
    """合并重叠或相接的区间,返回按时间排序的互不重叠区间。"""
    merged: list[tuple[datetime, datetime]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            # 与上一个区间重叠/相接:延长上一个区间的结束时间
            merged[-1] = merged[-1][0], max(merged[-1][1], end)
        else:
            merged.append((start, end))
    return merged


def _subtract(
    included: list[tuple[datetime, datetime]], excluded: list[tuple[datetime, datetime]]
) -> list[tuple[datetime, datetime]]:
    """从"纳入区间"中扣除"排除区间",返回剩余的区间片段。"""
    pieces = _union(included)
    for cut_start, cut_end in _union(excluded):
        next_pieces = []
        for start, end in pieces:
            if cut_end <= start or cut_start >= end:
                # 完全不相交:原样保留
                next_pieces.append((start, end))
            else:
                # 有交集:保留被切掉部分左右两侧剩余的片段
                if start < cut_start:
                    next_pieces.append((start, cut_start))
                if cut_end < end:
                    next_pieces.append((cut_end, end))
        pieces = next_pieces
    return pieces


def _minute_map(ranges: list[tuple[datetime, datetime]]) -> dict[str, int]:
    """把区间按日历日拆分,返回 {日期: 分钟数}。跨午夜的区间会分别记到两天。"""
    minutes: dict[str, int] = defaultdict(int)
    for start, end in ranges:
        cursor = start
        while cursor < end:
            # 当天结束点(次日零点)与区间结束点取较早者
            tomorrow = datetime.combine(cursor.date() + timedelta(days=1), time.min)
            stop = min(end, tomorrow)
            minutes[cursor.date().isoformat()] += int((stop - cursor).total_seconds() // 60)
            cursor = stop
    return dict(minutes)


def event_minutes(event: ResolvedEvent) -> list[dict[str, int]]:
    """返回一个事件所有"可能"的患者分钟数方案(每个方案是 {日期: 分钟});若有不确定性,则额外包含零分钟方案。"""
    # 未提供或患者未接受治疗:只有"零分钟"一种可能
    if event.disposition == "not_delivered" or event.patient_therapy == "no":
        return [{}]
    # 缺少日期或时间区间:无法量化,按零分钟处理(另由 unquantified 逻辑标记)
    if not event.service_date or not event.interval_options:
        return [{}]
    day = date.fromisoformat(event.service_date)
    excluded = _ranges(day, event.excluded_intervals)
    # 每个备选时间区间方案都扣除排除区间后计算分钟
    choices = [
        _minute_map(_subtract(_ranges(day, option), excluded))
        for option in event.interval_options
    ]
    # 事件本身存在不确定性时,"根本没发生"也是一种可能
    if event.disposition == "uncertain" or event.patient_therapy == "uncertain":
        choices.append({})
    return choices


def _therapy_type(service_type: str) -> str | None:
    """把服务类型归类为 individual/group/family/therapy_unspecified;非治疗类(如用药管理)返回 None。"""
    words = set(re.findall(r"[a-z]+", service_type.lower()))
    # 用药管理、协调、旁系信息、行政、外联等不算心理治疗
    if words & {"medication", "management", "coordination", "collateral", "administrative", "outreach"}:
        return None
    for category in ("individual", "group", "family"):
        if category in words:
            return category
    # 只写了"心理治疗/治疗"而未说明形式
    return "therapy_unspecified" if words & {"psychotherapy", "therapy"} else None


def _bounds(values: list[int]) -> dict[str, int]:
    """返回数值列表的最小/最大值;空列表时都是 0。"""
    return {"minimum": min(values, default=0), "maximum": max(values, default=0)}


def _week_start(day: date) -> date:
    """返回给定日期所在周的周一。"""
    return day - timedelta(days=day.weekday())


def _measure_instances(extraction: BatchExtraction) -> list[dict[str, Any]]:
    """把量表提及按"量表名 + 表单标识"归并为实例,标出日期/分数是否冲突以及复制来源。"""
    grouped: dict[tuple[str, str], list] = defaultdict(list)
    for item in extraction.measures:
        # 标识优先级:复制来源表单 > 表单编号 > 证据锚点引用
        # (复制出来的记录与原表单归为同一个实例,避免重复计数)
        identity = item.copied_from_form or item.form_id or item.anchor().reference()
        grouped[(item.instrument.strip().lower(), identity)].append(item)
    instances = []
    for (instrument, identity), claims in sorted(grouped.items()):
        dates = sorted({item.completed_date for item in claims if item.completed_date})
        scores = sorted({item.score for item in claims if item.score is not None})
        instances.append(
            {
                "instrument": instrument,
                "identity": identity,
                "completed_dates": dates,
                "scores": scores,
                # 同一实例出现多个日期或多个分数即视为冲突
                "conflicting_values": len(dates) > 1 or len(scores) > 1,
                "source_refs": [item.anchor().reference() for item in claims],
                "copy_refs": [item.anchor().reference() for item in claims if item.copied_from_form],
            }
        )
    return instances


def calculate_review(
    reconciliation: Reconciliation,
    extraction: BatchExtraction,
    period_start: str | None = None,
    period_end: str | None = None,
    findings: list[AuditFinding] | None = None,
) -> dict[str, Any]:
    """构建完整的事件台账和带上下界的汇总,全程不让模型做加法。"""
    first = date.fromisoformat(period_start) if period_start else None
    last = date.fromisoformat(period_end) if period_end else None
    mentions = {mention.mention_id: mention for mention in extraction.events}
    ledger: list[dict[str, Any]] = []  # 事件台账
    day_min: dict[str, int] = defaultdict(int)  # 每天的最少分钟数
    day_max: dict[str, int] = defaultdict(int)  # 每天的最多分钟数
    day_certain: set[str] = set()  # 确定有治疗的日期
    day_possible: set[str] = set()  # 可能有治疗的日期
    # 按类别统计的会话数,[0]=确定数(下界),[1]=可能数(上界)
    type_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    unquantified_event_ids: list[str] = []  # 无法量化(缺日期或时间)的事件
    # 覆盖缺口:存在疑似遗漏事件或时间范围不确定的审计发现
    coverage_gaps = [
        finding.detail for finding in (findings or [])
        if finding.code in {"possible_event_omission", "time_scope_uncertain"}
    ]

    for event in reconciliation.events:
        # 收集该事件所有相关来源的引用
        source_refs = sorted(
            {mentions[item].anchor().reference() for item in event.supporting_mentions + event.opposing_mentions if item in mentions}
        )
        options = event_minutes(event)
        date_totals = sorted(set(day for option in options for day in option))
        event_totals = [sum(option.values()) for option in options]
        category = _therapy_type(event.service_type)
        eligible = category is not None  # 是否为计入统计的治疗类服务
        # 事件是否落在统计周期内(无日期时保守地视为在周期内)
        within = not event.service_date or (
            (first is None or date.fromisoformat(event.service_date) >= first)
            and (last is None or date.fromisoformat(event.service_date) <= last)
        )
        if eligible and within:
            # 可能发生、但缺日期或时间区间:记为"无法量化"
            if event.disposition != "not_delivered" and event.patient_therapy != "no" and (not event.service_date or not event.interval_options):
                unquantified_event_ids.append(event.event_id)
            # 所有方案下分钟都 >0 → 确定计入;任一方案 >0 → 可能计入
            type_counts[category][0] += int(all(value > 0 for value in event_totals))
            type_counts[category][1] += int(any(value > 0 for value in event_totals))
            for day in date_totals:
                # 跨午夜拆分出的日期可能落在周期之外,跳过
                if first and date.fromisoformat(day) < first or last and date.fromisoformat(day) > last:
                    continue
                possible = [option.get(day, 0) for option in options]
                day_min[day] += min(possible)
                day_max[day] += max(possible)
                if min(possible) > 0:
                    day_certain.add(day)
                if max(possible) > 0:
                    day_possible.add(day)
        ledger.append(
            {
                "event_id": event.event_id,
                "service_date": event.service_date,
                "service_type": event.service_type,
                "service_category": category,
                "disposition": event.disposition,
                "patient_therapy": event.patient_therapy,
                "eligible_service": eligible,
                "within_period": within,
                "unquantified": event.event_id in unquantified_event_ids,
                "minute_options": options,
                "minutes": _bounds(event_totals),
                "source_refs": source_refs,
                "supporting_mentions": event.supporting_mentions,
                "opposing_mentions": event.opposing_mentions,
                "decision_notes": event.decision_notes,
            }
        )

    # 按周(周一至周日)归并有治疗的日期
    week_days: dict[str, list[str]] = defaultdict(list)
    for day in sorted(day_possible):
        week_days[_week_start(date.fromisoformat(day)).isoformat()].append(day)
    weeks: list[dict[str, Any]] = []
    # 指定了完整周期时,把没有任何治疗的周也补成空周,避免漏掉"零治疗周"
    if first and last:
        cursor = _week_start(first)
        while cursor <= last:
            week_days.setdefault(cursor.isoformat(), [])
            cursor += timedelta(days=7)
    for week, days in sorted(week_days.items()):
        monday = date.fromisoformat(week)
        weeks.append(
            {
                "week_start": week,
                "week_end": (monday + timedelta(days=6)).isoformat(),
                "days": {day: _bounds([day_min[day], day_max[day]]) for day in days},
                "therapy_days": {
                    "minimum": sum(day in day_certain for day in days),
                    "maximum": len(days),
                },
                "minutes": {
                    "minimum": sum(day_min[day] for day in days),
                    "maximum": sum(day_max[day] for day in days),
                },
            }
        )
    goals = [goal.model_dump() for goal in extraction.goals]
    # 逐周评估计划目标的达成情况
    for week in weeks:
        # 只评估边界明确为周一至周日、生效期与该周相交、且天数/分钟数都有明确数值的目标
        applicable = [
            goal for goal in extraction.goals
            if goal.period == "week_monday_sunday"
            and (not goal.effective_from or goal.effective_from <= week["week_end"])
            and (not goal.effective_to or goal.effective_to >= week["week_start"])
            and goal.minimum_days is not None and goal.minimum_minutes is not None
        ]
        week["goals"] = []
        for goal in applicable:
            day_bounds, minute_bounds = week["therapy_days"], week["minutes"]
            if day_bounds["minimum"] >= goal.minimum_days and minute_bounds["minimum"] >= goal.minimum_minutes:
                # 下界都已达标 → 确定达成
                status = "met"
            elif day_bounds["maximum"] < goal.minimum_days or minute_bounds["maximum"] < goal.minimum_minutes:
                # 上界都达不到 → 确定未达成
                status = "unmet"
            else:
                # 介于两者之间 → 无法判定
                status = "indeterminate"
            week["goals"].append(
                {
                    "source_ref": goal.anchor().reference(),
                    "minimum_days": goal.minimum_days,
                    "minimum_minutes": goal.minimum_minutes,
                    "status": status,
                }
            )
    return {
        "period": {"start": period_start, "end": period_end},
        "events": ledger,
        "sessions_by_type": {key: _bounds(values) for key, values in sorted(type_counts.items())},
        "therapy_sessions": {
            "minimum": sum(values[0] for values in type_counts.values()),
            "maximum": sum(values[1] for values in type_counts.values()),
        },
        "therapy_days": {"minimum": len(day_certain), "maximum": len(day_possible)},
        "therapy_minutes": {"minimum": sum(day_min.values()), "maximum": sum(day_max.values())},
        "unquantified_event_ids": unquantified_event_ids,
        "unresolved_mention_ids": reconciliation.unresolved_mention_ids,
        "coverage_gaps": coverage_gaps,
        # 只有没有无法量化事件、没有未归类提及、也没有覆盖缺口时,汇总才算完整
        "totals_complete": not unquantified_event_ids and not reconciliation.unresolved_mention_ids and not coverage_gaps,
        "weeks": weeks,
        "goals": goals,
        "measure_instances": _measure_instances(extraction),
    }
