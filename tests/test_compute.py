"""独立于开发用问题的领域级计算检查。"""

import unittest

from bb.compute import calculate_review, event_minutes
from bb.models import BatchExtraction, PlanGoal, Reconciliation, ResolvedEvent, TimeSpan


def span(start: str, end: str) -> TimeSpan:
    """快捷构造一个时间区间。"""
    return TimeSpan(start=start, end=end)


class CalculationTests(unittest.TestCase):
    def test_interval_union_exclusion_and_conflict(self) -> None:
        """区间应先取并集、再扣除排除区间;相互冲突的备选方案各自独立计算,不相加。"""
        event = ResolvedEvent(
            event_id="case:visit",
            service_date="2026-03-02",
            service_type="group psychotherapy",
            disposition="delivered",
            patient_therapy="yes",
            interval_options=[
                # 方案一:09:00-10:15 合并后共 75 分钟,扣除 09:30-09:45 的 15 分钟 → 60 分钟
                [span("09:00", "10:00"), span("09:45", "10:15")],
                # 方案二:09:10-10:15 共 65 分钟,扣除 15 分钟 → 50 分钟
                [span("09:10", "10:15")],
            ],
            excluded_intervals=[span("09:30", "09:45")],
        )
        self.assertEqual([sum(option.values()) for option in event_minutes(event)], [60, 50])

    def test_week_goal_remains_indeterminate_across_bounds(self) -> None:
        """当上下界分居目标两侧时,周目标状态应为"无法判定"。"""
        # 第一个事件:确定发生的个体治疗,60 分钟
        first = ResolvedEvent(
            event_id="case:one",
            service_date="2026-03-02",
            service_type="individual therapy",
            disposition="delivered",
            patient_therapy="yes",
            interval_options=[[span("09:00", "10:00")]],
        )
        # 第二个事件:是否发生都不确定的家庭治疗,可能 60 分钟,也可能 0 分钟
        second = ResolvedEvent(
            event_id="case:two",
            service_date="2026-03-04",
            service_type="family psychotherapy",
            disposition="uncertain",
            patient_therapy="uncertain",
            interval_options=[[span("09:00", "10:00")]],
        )
        # 目标:每周(周一至周日)至少 2 天、100 分钟
        goal = PlanGoal(source_id="PLAN", lines=[1], period="Monday-Sunday week", minimum_days=2, minimum_minutes=100)
        result = calculate_review(
            Reconciliation(events=[first, second]),
            BatchExtraction(goals=[goal]),
            "2026-03-02",
            "2026-03-08",
        )
        # 天数:下界 1、上界 2;分钟:下界 60、上界 120
        self.assertEqual(result["therapy_days"], {"minimum": 1, "maximum": 2})
        self.assertEqual(result["therapy_minutes"], {"minimum": 60, "maximum": 120})
        self.assertEqual(result["sessions_by_type"]["individual"], {"minimum": 1, "maximum": 1})
        self.assertEqual(result["sessions_by_type"]["family"], {"minimum": 0, "maximum": 1})
        # 下界达不到目标、上界又够得着目标 → 无法判定
        self.assertEqual(result["weeks"][0]["goals"][0]["status"], "indeterminate")


if __name__ == "__main__":
    unittest.main()
