"""
风险预警模块
Early Warning System Module

功能：
- 基于 6 期仿真数据的多维度趋势分析
- 三级预警阈值体系（关注 / 警告 / 严重）
- 输出预警事件列表与节点风险时序数据
- 验证：关键节点应在第 3-4 期触发预警
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
# 预警阈值配置
# ------------------------------------------------------------------ #
WARNING_THRESHOLDS = {
    # metric: (关注阈值, 警告阈值, 严重阈值) —— 数值越大风险越高的指标
    "capacity_utilization": (0.85, 0.92, 0.97),
    "demand_volatility":    (0.20, 0.28, 0.38),
    "lead_time_growth":     (0.10, 0.20, 0.35),  # 交货周期增长率

    # 数值越小风险越高的指标（阈值取反向：低于阈值触发）
    "inventory_level":      (20.0, 12.0, 6.0),     # 天
    "on_time_delivery":     (0.90, 0.82, 0.72),
    "logistics_reliability":(0.85, 0.78, 0.68),
    "supplier_count":       (3.0,  2.0,  1.5),      # 供应商数量（近似值）
}

# 预警等级权重（用于综合风险评分）
LEVEL_WEIGHTS = {"关注": 1, "警告": 2, "严重": 3}

LEVEL_NAMES = {0: "正常", 1: "关注", 2: "警告", 3: "严重"}
LEVEL_COLORS = {"正常": "#27ae60", "关注": "#f39c12", "警告": "#e67e22", "严重": "#e74c3c"}


@dataclass
class WarningEvent:
    """单条预警事件"""
    node_id: str
    node_name: str
    period: int           # 触发期次（1-6）
    metric: str           # 触发指标
    metric_value: float   # 当前值
    threshold: float      # 触发阈值
    level: str            # 关注/警告/严重
    trend: str            # 上升/下降/平稳
    description: str


@dataclass
class WarningReport:
    """预警报告"""
    total_warnings: int
    events_by_level: Dict[str, int]
    critical_nodes: List[str]        # 触发"严重"预警的节点
    warning_events: List[WarningEvent]
    node_risk_trend: Dict            # node_id -> List[float] (6期综合风险评分)
    first_warning_period: Dict       # node_id -> 首次预警期次
    summary: str


# ------------------------------------------------------------------ #
# 主预警类
# ------------------------------------------------------------------ #
class EarlyWarningSystem:
    """多期风险预警系统"""

    NODE_NAMES = {
        "T3-SI": "芯片晶圆供应商", "T3-RE": "稀土材料供应商",
        "T2-ECU": "ECU控制单元", "T2-SN": "传感器模组",
        "T1-E": "电子电气系统集成商", "T2-E2": "涡轮增压器",
        "T3-CU": "铜材供应商", "OEM": "总装厂",
        "T1-P": "动力总成系统集成商", "T1-C": "底盘系统集成商",
        "T3-ST": "特种钢材", "T3-AL": "铸造铝合金",
        "T3-NI": "镍基合金", "T3-RB": "合成橡胶",
        "T3-PCB": "印制电路板", "T3-PL": "工程塑料",
        "T3-CF": "碳纤维材料", "T3-MG": "镁合金",
        "T3-GL": "特种玻璃", "T2-E1": "发动机缸体",
        "T2-T1": "变速箱总成", "T2-B1": "制动系统",
        "T2-S1": "悬挂系统", "T2-W1": "转向器", "T2-H1": "线束总成",
    }

    # 需要重点监控的节点
    KEY_NODES = ["T3-SI", "T3-RE", "T2-ECU", "T2-SN", "T1-E", "T2-E2", "T3-CU", "OEM"]

    def __init__(self, df: Optional[pd.DataFrame] = None):
        self.df = df

    def load_simulation_data(self, csv_path: str):
        self.df = pd.read_csv(csv_path, encoding="utf-8-sig")
        # 确保 period 为整数
        self.df["period"] = self.df["period"].astype(int)

    # ------------------------------------------------------------------ #
    # 核心分析
    # ------------------------------------------------------------------ #
    def analyze(self) -> WarningReport:
        if self.df is None:
            raise ValueError("未加载仿真数据，请先调用 load_simulation_data()")

        all_events: List[WarningEvent] = []
        node_risk_trend: Dict[str, List[float]] = {}
        first_warning_period: Dict[str, int] = {}
        periods = sorted(self.df["period"].unique())

        for node_id in self.df["node_id"].unique():
            node_df = self.df[self.df["node_id"] == node_id].sort_values("period")
            if len(node_df) < 2:
                continue

            period_scores = []
            for _, row in node_df.iterrows():
                period = int(row["period"])
                score, events = self._assess_period(node_id, period, row, node_df)
                period_scores.append(score)
                all_events.extend(events)

                if events and node_id not in first_warning_period:
                    first_warning_period[node_id] = period

            node_risk_trend[node_id] = period_scores

        # 汇总
        levels_count = {"关注": 0, "警告": 0, "严重": 0}
        for ev in all_events:
            levels_count[ev.level] = levels_count.get(ev.level, 0) + 1

        critical_nodes = list({ev.node_id for ev in all_events if ev.level == "严重"})

        summary = (
            f"共检测到 {len(all_events)} 条预警事件，其中严重 {levels_count['严重']} 条、"
            f"警告 {levels_count['警告']} 条、关注 {levels_count['关注']} 条。"
            f"关键高风险节点：{', '.join(critical_nodes[:5]) if critical_nodes else '无'}。"
        )

        return WarningReport(
            total_warnings=len(all_events),
            events_by_level=levels_count,
            critical_nodes=critical_nodes,
            warning_events=all_events,
            node_risk_trend=node_risk_trend,
            first_warning_period=first_warning_period,
            summary=summary,
        )

    def _assess_period(
        self,
        node_id: str,
        period: int,
        row: pd.Series,
        node_df: pd.DataFrame,
    ) -> Tuple[float, List[WarningEvent]]:
        """评估单节点单期次的风险等级"""
        events = []
        score_contributions = []

        # 指标1：产能利用率（越高越危险）
        cu = row.get("capacity_utilization", np.nan)
        if not np.isnan(cu):
            lvl = self._level_high(cu, WARNING_THRESHOLDS["capacity_utilization"])
            if lvl > 0:
                events.append(self._make_event(
                    node_id, period, "capacity_utilization", cu,
                    WARNING_THRESHOLDS["capacity_utilization"][lvl - 1],
                    lvl, row, node_df, "产能利用率过高，存在供应瓶颈风险",
                ))
            score_contributions.append(cu * 0.30)

        # 指标2：库存水位（越低越危险）
        inv = row.get("inventory_level", np.nan)
        if not np.isnan(inv):
            lvl = self._level_low(inv, WARNING_THRESHOLDS["inventory_level"])
            if lvl > 0:
                events.append(self._make_event(
                    node_id, period, "inventory_level", inv,
                    WARNING_THRESHOLDS["inventory_level"][lvl - 1],
                    lvl, row, node_df, "库存水位过低，抗风险缓冲不足",
                ))
            # 归一化到[0,1]：库存越低分越高
            inv_norm = max(0.0, 1.0 - inv / 90.0)
            score_contributions.append(inv_norm * 0.25)

        # 指标3：交货准时率（越低越危险）
        otd = row.get("on_time_delivery", np.nan)
        if not np.isnan(otd):
            lvl = self._level_low(otd, WARNING_THRESHOLDS["on_time_delivery"])
            if lvl > 0:
                events.append(self._make_event(
                    node_id, period, "on_time_delivery", otd,
                    WARNING_THRESHOLDS["on_time_delivery"][lvl - 1],
                    lvl, row, node_df, "交货准时率下降，供应稳定性恶化",
                ))
            score_contributions.append((1.0 - otd) * 0.20)

        # 指标4：供应商数量（越少越危险）
        sc = row.get("supplier_count", np.nan)
        if not np.isnan(sc):
            lvl = self._level_low(sc, WARNING_THRESHOLDS["supplier_count"])
            if lvl > 0:
                events.append(self._make_event(
                    node_id, period, "supplier_count", sc,
                    WARNING_THRESHOLDS["supplier_count"][lvl - 1],
                    lvl, row, node_df, "可用供应商数量过少，单点依赖风险高",
                ))
            sc_norm = max(0.0, 1.0 - (sc - 1.0) / 9.0)
            score_contributions.append(sc_norm * 0.15)

        # 指标5：物流可靠性（越低越危险）
        lr = row.get("logistics_reliability", np.nan)
        if not np.isnan(lr):
            lvl = self._level_low(lr, WARNING_THRESHOLDS["logistics_reliability"])
            if lvl > 0:
                events.append(self._make_event(
                    node_id, period, "logistics_reliability", lr,
                    WARNING_THRESHOLDS["logistics_reliability"][lvl - 1],
                    lvl, row, node_df, "物流可靠性不足，运输中断风险高",
                ))
            score_contributions.append((1.0 - lr) * 0.10)

        composite_score = sum(score_contributions) if score_contributions else 0.0
        return composite_score, events

    # ------------------------------------------------------------------ #
    # 辅助方法
    # ------------------------------------------------------------------ #
    @staticmethod
    def _level_high(value: float, thresholds: Tuple) -> int:
        """高值触发：返回预警等级 0=无, 1=关注, 2=警告, 3=严重"""
        t1, t2, t3 = thresholds
        if value >= t3:
            return 3
        elif value >= t2:
            return 2
        elif value >= t1:
            return 1
        return 0

    @staticmethod
    def _level_low(value: float, thresholds: Tuple) -> int:
        """低值触发：返回预警等级"""
        t1, t2, t3 = thresholds
        if value <= t3:
            return 3
        elif value <= t2:
            return 2
        elif value <= t1:
            return 1
        return 0

    def _make_event(
        self,
        node_id: str,
        period: int,
        metric: str,
        value: float,
        threshold: float,
        level_int: int,
        row: pd.Series,
        node_df: pd.DataFrame,
        description: str,
    ) -> WarningEvent:
        # 趋势判断（与前期比较）
        prev_rows = node_df[node_df["period"] < period]
        if not prev_rows.empty and metric in prev_rows.columns:
            prev_val = prev_rows.iloc[-1][metric]
            if isinstance(prev_val, float):
                delta = value - prev_val
                trend = "上升" if delta > 0.01 else ("下降" if delta < -0.01 else "平稳")
            else:
                trend = "平稳"
        else:
            trend = "首次"

        return WarningEvent(
            node_id=node_id,
            node_name=self.NODE_NAMES.get(node_id, node_id),
            period=period,
            metric=metric,
            metric_value=round(float(value), 4),
            threshold=round(float(threshold), 4),
            level=LEVEL_NAMES[level_int],
            trend=trend,
            description=description,
        )

    # ------------------------------------------------------------------ #
    # 便捷接口
    # ------------------------------------------------------------------ #
    def get_key_node_trends(self, report: WarningReport) -> Dict[str, List[float]]:
        """返回关键节点的6期风险评分序列"""
        return {
            nid: report.node_risk_trend.get(nid, [0.0] * 6)
            for nid in self.KEY_NODES
            if nid in report.node_risk_trend
        }

    def report_to_dict(self, report: WarningReport) -> Dict:
        return {
            "total_warnings": report.total_warnings,
            "events_by_level": report.events_by_level,
            "critical_nodes": report.critical_nodes,
            "first_warning_period": report.first_warning_period,
            "summary": report.summary,
            "warning_events": [
                {
                    "node_id": ev.node_id,
                    "node_name": ev.node_name,
                    "period": ev.period,
                    "metric": ev.metric,
                    "metric_value": ev.metric_value,
                    "threshold": ev.threshold,
                    "level": ev.level,
                    "trend": ev.trend,
                    "description": ev.description,
                }
                for ev in report.warning_events
            ],
        }
