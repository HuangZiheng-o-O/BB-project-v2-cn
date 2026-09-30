"""对"有来源依据的模型阶段输出"做有次数上限的修复。"""

from __future__ import annotations

import json
from typing import Any, Callable

from bb.model_provider import ModelPort, generate_json


class StageValidationError(ValueError):
    """模型阶段在经过反馈和多次重新生成后,输出仍然无效。"""


def generate_checked_json(
    model: ModelPort,
    system: str,
    source_prompt: str,
    validate: Callable[[dict[str, Any]], list[str]],
    *,
    max_tokens: int,
    attempts: int = 3,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """返回经过完整校验的结果;若始终无效,则在下游阶段使用它之前就报错终止。

    领域规则由 validator 负责。修复循环只负责把错误反馈给模型,并基于原始证据重试,
    因此某一个问题无法改变规则本身。
    """
    trace: list[dict[str, Any]] = []
    feedback = ""  # 上一轮失败后拼接到提示词末尾的反馈
    for attempt in range(1, attempts + 1):
        prompt = source_prompt + feedback
        try:
            result, calls = generate_json(model, system, prompt, max_tokens=max_tokens)
        except ValueError as error:
            # 模型没给出可解析的 JSON:把它当作一条校验错误处理
            errors = [str(error)]
            result, calls = {}, []
        else:
            errors = validate(result)
        trace.extend({**call, "validation_attempt": attempt} for call in calls)
        trace.append({"stage": "validation", "validation_attempt": attempt, "errors": errors})
        # 没有错误就立即返回
        if not errors:
            return result, trace
        previous = json.dumps(result, ensure_ascii=False)
        # 构造下一轮的反馈:要求重读原始证据、补全修正,最多带 30 条错误;
        # 上一轮候选结果过长(>30000 字符)时不再附带,避免提示词膨胀
        feedback = (
            "\n\nThe previous candidate failed validation. Re-read the original evidence "
            "and repair the candidate into a complete corrected JSON object. Do not omit valid claims "
            "or invent missing evidence. Validation errors:\n"
            + json.dumps(errors[:30], ensure_ascii=False)
            + (f"\nAdditional errors: {len(errors) - 30}." if len(errors) > 30 else "")
            + (f"\nPrevious candidate:\n{previous}" if len(previous) <= 30000 else "")
        )
    # 所有尝试都失败:抛出异常,附上前 5 条错误
    raise StageValidationError(
        f"Model stage failed validation after {attempts} attempts: "
        + "; ".join(errors[:5])
    )
