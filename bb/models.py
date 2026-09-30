"""抽取、对账(reconcile)与审查各阶段共享的、带版本的数据契约(Pydantic 模型)。"""

from __future__ import annotations

from datetime import date
from hashlib import sha256
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class TimeSpan(BaseModel):
    """一段时间区间,使用当地 24 小时制 HH:MM。"""

    start: str = Field(description="Local HH:MM start time")
    end: str = Field(description="Local HH:MM end time")

    @field_validator("start", "end")
    @classmethod
    def valid_clock(cls, value: str) -> str:
        # 只接受 00:00 ~ 23:59 的 HH:MM 格式
        if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value):
            raise ValueError("Time must use 24-hour HH:MM")
        return value


def _valid_date(value: str | None) -> str | None:
    """校验日期:None 直接放行;否则必须是 YYYY-MM-DD 且是真实存在的日期。"""
    if value is not None:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError("Date must use YYYY-MM-DD")
        # fromisoformat 会拒绝 2026-02-30 这类不存在的日期
        date.fromisoformat(value)
    return value


def _empty_string(value: str | None) -> str:
    """把模型可能返回的 None 规范化为空字符串。"""
    return "" if value is None else value


def _empty_list(value: list | None) -> list:
    """把模型可能返回的 None 规范化为空列表。"""
    return [] if value is None else value


def _source_lines(value: object) -> object:
    """接受带编号的来源行标签(如 "L12"、"L12-L14"、"L1,L3"),统一存成整数行号列表。"""
    if not isinstance(value, list):
        return value
    numbers: list[int] = []
    for entry in value:
        # 整数直接采纳(bool 是 int 的子类,需要排除)
        if isinstance(entry, int) and not isinstance(entry, bool):
            numbers.append(entry)
            continue
        if not isinstance(entry, str):
            raise ValueError("Source lines must be integers or L-prefixed line labels")
        # 一个字符串里可能包含逗号分隔的多个片段
        for part in entry.split(","):
            # 匹配 "L12"、"12"、"L12-L14"、"12–14" 等写法
            match = re.fullmatch(r"L?(\d+)(?:[-–]L?(\d+))?", part.strip())
            if not match:
                raise ValueError(f"Invalid source line label: {entry}")
            first = int(match.group(1))
            last = int(match.group(2)) if match.group(2) else first
            # 区间不能倒置,也不能超过 12 行(防止模型引用过大范围)
            if last < first or last - first >= 12:
                raise ValueError(f"Invalid or oversized source line range: {entry}")
            numbers.extend(range(first, last + 1))
    return numbers


class Anchor(BaseModel):
    """证据锚点:指向某个来源文档中的若干行。"""

    source_id: str
    lines: list[int] = Field(min_length=1, max_length=12)

    @field_validator("lines", mode="before")
    @classmethod
    def parse_lines(cls, value: object) -> object:
        # 校验前先把各种行号写法解析成整数列表
        return _source_lines(value)

    @field_validator("lines")
    @classmethod
    def positive_lines(cls, values: list[int]) -> list[int]:
        if any(value < 1 for value in values):
            raise ValueError("Source line numbers must be positive")
        # 去重并排序,得到规范形式
        return sorted(set(values))

    def reference(self) -> str:
        """生成引用字符串,形如 "SRC-01:L3,L4"。"""
        numbers = ",".join(f"L{line}" for line in self.lines)
        return f"{self.source_id}:{numbers}"


class EventMention(BaseModel):
    """某个来源文档对一次诊疗事件的"提及"——记录该来源声称了什么,而非最终对账结论。"""

    source_id: str
    lines: list[int] = Field(min_length=1, max_length=12)
    patient_id: str | None = None
    encounter_id: str | None = None
    appointment_id: str | None = None
    service_date: str | None = None
    service_type: str | None = None
    document_role: str = Field(description="clinical, attendance, schedule, correction, charge, draft, or administrative")
    status: str = Field(description="What this source asserts, not the reconciled event status")
    patient_present: bool | None = None
    actual_intervals: list[TimeSpan] = Field(default_factory=list)  # 实际发生的时间段
    scheduled_intervals: list[TimeSpan] = Field(default_factory=list)  # 计划/预约的时间段
    nontherapy_intervals: list[TimeSpan] = Field(default_factory=list)  # 非治疗时间段(如休息)
    correction_field: str | None = None  # 更正记录所更正的字段
    correction_value: str | None = None  # 更正后的值
    duplicate_of: str | None = None  # 若是重复记录,指向原记录
    note: str = ""
    mention_id: str = ""  # 由 finalize_id 生成的稳定标识

    @field_validator("lines", mode="before")
    @classmethod
    def parse_lines(cls, value: object) -> object:
        return _source_lines(value)

    @field_validator("note", mode="before")
    @classmethod
    def optional_note(cls, value: str | None) -> str:
        return _empty_string(value)

    @field_validator("actual_intervals", "scheduled_intervals", "nontherapy_intervals", mode="before")
    @classmethod
    def optional_intervals(cls, value: list | None) -> list:
        return _empty_list(value)

    @field_validator("service_date")
    @classmethod
    def valid_service_date(cls, value: str | None) -> str | None:
        return _valid_date(value)

    def finalize_id(self) -> None:
        """根据来源、患者、就诊、日期和行号生成 16 位十六进制的确定性 ID。"""
        key = f"{self.source_id}|{self.patient_id}|{self.encounter_id}|{self.service_date}|{self.lines}"
        self.mention_id = sha256(key.encode()).hexdigest()[:16]

    def anchor(self) -> Anchor:
        """返回该提及对应的证据锚点。"""
        return Anchor(source_id=self.source_id, lines=self.lines)


class PlanGoal(BaseModel):
    """治疗计划中的目标(如每周最少治疗天数/分钟数)。"""

    source_id: str
    lines: list[int] = Field(min_length=1, max_length=12)
    effective_from: str | None = None  # 生效起始日期
    effective_to: str | None = None  # 生效截止日期
    period: str | None = None  # 统计周期(如周一至周日)
    minimum_days: int | None = None  # 周期内最少天数
    minimum_minutes: int | None = None  # 周期内最少分钟数
    included_services: list[str] = Field(default_factory=list)  # 计入目标的服务类型
    excluded_services: list[str] = Field(default_factory=list)  # 不计入目标的服务类型
    description: str = ""

    @field_validator("lines", mode="before")
    @classmethod
    def parse_lines(cls, value: object) -> object:
        return _source_lines(value)

    @field_validator("period", mode="before")
    @classmethod
    def normalize_period(cls, value: str | None) -> str | None:
        """把模型给出的各种周期表述归一化。"""
        if value is None:
            return None
        lowered = value.lower().strip()
        # 明确写了"周一到周日"的,归一为固定的周边界
        if lowered == "week_monday_sunday" or ("monday" in lowered and "sunday" in lowered):
            return "week_monday_sunday"
        # 只说"每周"但没说边界的,单独标记,后续不会当作周一至周日来计算
        if lowered in {"weekly", "week", "each week"}:
            return "weekly_unspecified_boundary"
        return value

    @field_validator("included_services", "excluded_services", mode="before")
    @classmethod
    def optional_services(cls, value: list | None) -> list:
        return _empty_list(value)

    @field_validator("description", mode="before")
    @classmethod
    def optional_description(cls, value: str | None) -> str:
        return _empty_string(value)

    @field_validator("effective_from", "effective_to")
    @classmethod
    def valid_effective_date(cls, value: str | None) -> str | None:
        return _valid_date(value)

    def anchor(self) -> Anchor:
        return Anchor(source_id=self.source_id, lines=self.lines)


class MeasureMention(BaseModel):
    """对症状量表(如评估问卷)的一次提及。"""

    source_id: str
    lines: list[int] = Field(min_length=1, max_length=12)
    instrument: str  # 量表名称
    form_id: str | None = None  # 表单编号
    completed_date: str | None = None  # 完成日期
    score: float | None = None  # 得分
    copied_from_form: str | None = None  # 若是从别的表单复制而来,记录原表单编号
    note: str = ""

    @field_validator("lines", mode="before")
    @classmethod
    def parse_lines(cls, value: object) -> object:
        return _source_lines(value)

    @field_validator("note", mode="before")
    @classmethod
    def optional_note(cls, value: str | None) -> str:
        return _empty_string(value)

    @field_validator("completed_date")
    @classmethod
    def valid_completed_date(cls, value: str | None) -> str | None:
        return _valid_date(value)

    def anchor(self) -> Anchor:
        return Anchor(source_id=self.source_id, lines=self.lines)


class Observation(BaseModel):
    """临床观察条目:围绕某个主题的一条带极性的陈述。"""

    source_id: str
    lines: list[int] = Field(min_length=1, max_length=12)
    date: str | None = None
    subject: str = "patient"  # 观察对象,默认是患者
    theme: str  # 主题
    statement: str  # 陈述内容
    # 极性:阳性 / 阴性 / 不确定 / 仅为计划
    polarity: Literal["positive", "negative", "uncertain", "planned"] = "positive"

    @field_validator("lines", mode="before")
    @classmethod
    def parse_lines(cls, value: object) -> object:
        return _source_lines(value)

    @field_validator("date")
    @classmethod
    def valid_observation_date(cls, value: str | None) -> str | None:
        return _valid_date(value)

    def anchor(self) -> Anchor:
        return Anchor(source_id=self.source_id, lines=self.lines)


class BatchExtraction(BaseModel):
    """一批(或全部)文档抽取结果的汇总:事件、目标、量表、观察。"""

    events: list[EventMention] = Field(default_factory=list)
    goals: list[PlanGoal] = Field(default_factory=list)
    measures: list[MeasureMention] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)


class ResolvedEvent(BaseModel):
    """对账之后得到的一次诊疗事件的最终结论。"""

    event_id: str
    service_date: str | None = None
    service_type: str
    # 是否实际提供:已提供 / 未提供 / 不确定
    disposition: Literal["delivered", "not_delivered", "uncertain"]
    # 患者是否实际参与了治疗:是 / 否 / 不确定
    patient_therapy: Literal["yes", "no", "uncertain"]
    # 可能的时间区间方案(证据冲突时可能有多个备选)
    interval_options: list[list[TimeSpan]] = Field(default_factory=list)
    # 需要从治疗时间中扣除的区间
    excluded_intervals: list[TimeSpan] = Field(default_factory=list)
    supporting_mentions: list[str] = Field(default_factory=list)  # 支持该结论的提及 ID
    opposing_mentions: list[str] = Field(default_factory=list)  # 与该结论冲突的提及 ID
    decision_notes: str = ""

    @field_validator("service_date")
    @classmethod
    def valid_service_date(cls, value: str | None) -> str | None:
        return _valid_date(value)


class Reconciliation(BaseModel):
    """对账阶段的整体输出。"""

    events: list[ResolvedEvent] = Field(default_factory=list)
    unresolved_mention_ids: list[str] = Field(default_factory=list)  # 未能归入任何事件的提及


class AuditFinding(BaseModel):
    """审计发现:某个阶段检查出的问题或不确定性。"""

    code: str  # 问题类型代码
    detail: str  # 详细说明
    source_refs: list[str] = Field(default_factory=list)


class ReviewSnapshot(BaseModel):
    """一次完整审查的快照,可落盘并在文档未变时复用。"""

    source_hashes: dict[str, str]  # 各来源文档的内容哈希
    extraction_model: str  # 产出抽取结果的模型名
    extraction: BatchExtraction
    reconciliation: Reconciliation
    findings: list[AuditFinding] = Field(default_factory=list)


# 这些审计代码表示模型阶段的输出被拒绝或有遗漏,带有它们的快照不允许复用
INVALID_STAGE_FINDINGS = {
    "invalid_extraction_array",
    "invalid_source_candidate",
    "possible_event_omission",
    "invalid_event_decision",
    "unresolved_event_group",
}


def validate_snapshot_reuse(
    snapshot: ReviewSnapshot, model_name: str, source_hashes: dict[str, str],
) -> None:
    """确保过期或被拒绝的模型阶段输出不会混入新的回答运行中。"""
    # 模型必须一致,否则抽取结果不可比
    if snapshot.extraction_model != model_name:
        raise ValueError("The abstraction was produced by a different model")
    # 文档内容必须未变
    if snapshot.source_hashes != source_hashes:
        raise ValueError("Snapshot source hashes do not match the document directory")
    # 快照中不能含有"被拒绝/有遗漏"类审计发现
    rejected = sorted({item.code for item in snapshot.findings if item.code in INVALID_STAGE_FINDINGS})
    if rejected:
        raise ValueError(f"The abstraction contains rejected or omitted source claims: {', '.join(rejected)}")
