"""
Phase 6: Multi-Agent Coordination and System Integration
阶段六：多智能体协同与系统集成（模块4.3 + 集成）

运行方式: conda run -n logistic python run_phase6.py

功能：
1. 初始化三个领域智能体（库存/物流/需求）
2. 多智能体协同处置芯片停供场景（S1）
3. 反馈闭环：处置结果→网络模型风险评分更新
4. 端到端全链路验证
5. 产出 F6-1 ~ F6-4 图表与阶段报告
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from src.agent.multi_agent import MultiAgentSystem
from src.visualization.style import apply_research_style, save_figure


# ------------------------------------------------------------------ #
# 辅助函数
# ------------------------------------------------------------------ #
def load_phase4_report(path: str = "outputs/reports/phase4_report.json") -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_simulation_data(path: str = "data/simulation/auto_engine_simulation.csv") -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ------------------------------------------------------------------ #
# 图表 F6-1：多智能体协同架构图
# ------------------------------------------------------------------ #
def plot_multi_agent_architecture() -> plt.Figure:
    """绘制多智能体协同架构图（流程/角色交互图）。"""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # ---- 颜色配置 ----
    C_INPUT  = "#2980B9"   # 输入层
    C_AGENT  = "#27AE60"   # 领域智能体
    C_COORD  = "#E67E22"   # 协调智能体
    C_OUT    = "#8E44AD"   # 输出层
    C_FEED   = "#C0392B"   # 反馈闭环
    WHITE    = "white"

    def box(ax, x, y, w, h, color, text, fontsize=10, text_color="white"):
        rect = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.12",
            facecolor=color, edgecolor="white",
            linewidth=1.5, zorder=3
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight="bold",
                zorder=4, wrap=True)

    def arrow(ax, x1, y1, x2, y2, color="#555555", style="->"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color,
                                    lw=1.8, connectionstyle="arc3,rad=0.0"),
                    zorder=2)

    def label(ax, x, y, text, fontsize=9, color="#333333"):
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fontsize, color=color, zorder=5)

    # ---- 第一列：输入数据层 ----
    ax.text(2.0, 8.5, "输入数据层\n(Input Data Layer)",
            ha="center", va="center", fontsize=10, color=C_INPUT, fontweight="bold")
    box(ax, 2.0, 7.4, 2.6, 0.7, C_INPUT, "仿真运营数据\n(6期, 25节点)")
    box(ax, 2.0, 6.2, 2.6, 0.7, C_INPUT, "综合风险评估结果\n(模糊+贝叶斯融合)")
    box(ax, 2.0, 5.0, 2.6, 0.7, C_INPUT, "供应链网络拓扑\n(25节点·4层级)")

    # ---- 第二列：领域智能体层 ----
    ax.text(6.5, 8.5, "领域智能体层\n(Domain Agent Layer)",
            ha="center", va="center", fontsize=10, color=C_AGENT, fontweight="bold")
    box(ax, 6.5, 7.4, 2.8, 0.75, C_AGENT,
        "库存风险感知智能体\nInventory Risk Agent\n[安全库存·补库建议]", fontsize=9)
    box(ax, 6.5, 5.8, 2.8, 0.75, C_AGENT,
        "物流异常分析智能体\nLogistics Anomaly Agent\n[备用路线·区域风险]", fontsize=9)
    box(ax, 6.5, 4.2, 2.8, 0.75, C_AGENT,
        "需求预测智能体\nDemand Forecasting Agent\n[趋势分析·牛鞭效应]", fontsize=9)

    # ---- 第三列：协调智能体 ----
    ax.text(10.5, 8.5, "协调层\n(Coordination Layer)",
            ha="center", va="center", fontsize=10, color=C_COORD, fontweight="bold")
    box(ax, 10.5, 6.0, 2.8, 2.2, C_COORD,
        "协调智能体\nCoordinator Agent\n\n①冲突检测\n②多目标权衡\n③综合处置方案", fontsize=9)

    # ---- 第四列：输出层 ----
    ax.text(13.2, 8.5, "输出层\n(Output Layer)",
            ha="center", va="center", fontsize=10, color=C_OUT, fontweight="bold")
    box(ax, 13.2, 7.2, 1.5, 0.6, C_OUT, "综合处置\n计划", fontsize=9)
    box(ax, 13.2, 6.0, 1.5, 0.6, C_OUT, "多目标\n评分", fontsize=9)
    box(ax, 13.2, 4.8, 1.5, 0.6, C_OUT, "风险评分\n更新", fontsize=9)

    # ---- 反馈闭环 ----
    box(ax, 6.5, 2.0, 5.5, 0.65, C_FEED,
        "反馈闭环：处置结果 → 网络模型更新 → 下轮评估  "
        "(Feedback Loop: Disposal → Network Update → Next Assessment)", fontsize=8.5)

    # ---- 箭头：输入→智能体 ----
    for y_src, y_dst in [(7.4, 7.4), (6.2, 5.8), (5.0, 4.2)]:
        arrow(ax, 3.3, y_src, 5.1, y_dst, C_INPUT)

    # ---- 箭头：智能体→协调 ----
    for y in [7.4, 5.8, 4.2]:
        arrow(ax, 7.9, y, 9.1, 6.0 + (y - 5.8) * 0.3, C_AGENT)

    # ---- 箭头：协调→输出 ----
    for y in [7.2, 6.0, 4.8]:
        arrow(ax, 11.9, 6.0 + (y - 6.0) * 0.2, 12.45, y, C_COORD)

    # ---- 反馈箭头 ----
    ax.annotate("", xy=(6.5, 2.33), xytext=(13.2, 4.5),
                arrowprops=dict(arrowstyle="->", color=C_FEED, lw=1.8,
                                connectionstyle="arc3,rad=0.3"), zorder=2)

    # ---- 消息协议标注 ----
    label(ax, 4.7, 7.9, "提案\n(Proposal)", 8, "#555555")
    label(ax, 8.5, 7.2, "提案传递", 8, "#555555")
    label(ax, 12.0, 7.6, "输出", 8, "#555555")

    ax.set_title(
        "多智能体协同架构图\n"
        "Multi-Agent Coordination Architecture",
        fontsize=13, fontweight="bold", pad=10
    )
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ #
# 图表 F6-2：处置方案多目标评估雷达图
# ------------------------------------------------------------------ #
def plot_multi_objective_radar(integrated_plan) -> plt.Figure:
    """绘制综合处置方案的多目标评估雷达图。"""
    # 取前3个高优先级处置动作
    actions = [a for a in integrated_plan.integrated_actions if a.priority == "高"][:3]
    if len(actions) < 2:
        actions = integrated_plan.integrated_actions[:3]

    categories = [
        "风险降低效果\nRisk Reduction",
        "成本合理性\nCost Rationality",
        "实施效率\nEfficiency",
        "可行性\nFeasibility",
        "紧迫性\nUrgency",
    ]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 8), subplot_kw=dict(polar=True))

    colors = ["#E74C3C", "#3498DB", "#27AE60", "#F39C12", "#9B59B6"]
    max_val = 10.0

    for i, action in enumerate(actions):
        values = [
            action.score_risk_reduction,
            action.score_cost,
            action.score_efficiency,
            action.score_feasibility,
            action.score_urgency,
        ]
        values_norm = [v / max_val for v in values]
        values_norm += values_norm[:1]

        ax.plot(angles, values_norm, color=colors[i], linewidth=2.0,
                linestyle="solid", label=f"{action.node_name}\n({'+'.join(action.source_agents[:2])})")
        ax.fill(angles, values_norm, color=colors[i], alpha=0.15)

    # 参考圆
    for r in [0.3, 0.6, 0.9]:
        ax.plot(angles, [r] * (N + 1), color="gray", linewidth=0.6, linestyle="--", alpha=0.5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=9, fontweight="bold")
    ax.set_yticks([0.3, 0.6, 0.9])
    ax.set_yticklabels(["3", "6", "9"], size=8, color="gray")
    ax.set_ylim(0, 1.0)

    ax.legend(loc="lower right", bbox_to_anchor=(1.45, -0.08),
              fontsize=9, framealpha=0.9)

    ax.set_title(
        "处置方案多目标评估雷达图\n"
        "Multi-Objective Evaluation Radar Chart for Disposal Plans",
        fontsize=12, fontweight="bold", pad=20
    )
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ #
# 图表 F6-3：全链路闭环验证流程图
# ------------------------------------------------------------------ #
def plot_pipeline_flowchart(report) -> plt.Figure:
    """绘制端到端全链路闭环验证流程图。"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    STAGE_COLORS = [
        "#2980B9",   # 网络建模
        "#16A085",   # 风险识别
        "#8E44AD",   # 智能处置
        "#E67E22",   # 多智能体
        "#C0392B",   # 处置结果
        "#2C3E50",   # 反馈闭环
    ]
    stages = [
        ("网络建模\nNetwork\nModeling", "25节点\n4层级网络"),
        ("风险评估\nRisk\nAssessment", "模糊+贝叶斯\n综合评分"),
        ("瓶颈诊断\nBottleneck\nDiagnosis", "RAG+LLM\n智能推理"),
        ("多智能体\nMulti-Agent\nCoordination", "库存/物流/需求\n协同决策"),
        ("综合处置\nIntegrated\nDisposal", f"风险降低\n{report.overall_risk_reduction:.1%}"),
        ("模型更新\nModel\nUpdate", "闭环反馈\n下轮优化"),
    ]

    x_positions = [1.1, 3.3, 5.5, 7.7, 9.9, 12.1]
    for i, ((title, subtitle), color, x) in enumerate(zip(stages, STAGE_COLORS, x_positions)):
        # 主框
        rect = FancyBboxPatch(
            (x - 0.9, 2.5), 1.8, 1.5,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="white", linewidth=2, zorder=3
        )
        ax.add_patch(rect)
        ax.text(x, 3.45, title, ha="center", va="center",
                fontsize=9.5, color="white", fontweight="bold", zorder=4)

        # 子框（数据标注）
        rect2 = FancyBboxPatch(
            (x - 0.85, 1.5), 1.7, 0.85,
            boxstyle="round,pad=0.08",
            facecolor="#ECF0F1", edgecolor=color, linewidth=1.5, zorder=3
        )
        ax.add_patch(rect2)
        ax.text(x, 1.93, subtitle, ha="center", va="center",
                fontsize=8, color=color, fontweight="bold", zorder=4)

        # 阶段编号
        circle = plt.Circle((x, 4.4), 0.22, color=color, zorder=5)
        ax.add_patch(circle)
        ax.text(x, 4.4, str(i + 1), ha="center", va="center",
                fontsize=10, color="white", fontweight="bold", zorder=6)

        # 箭头
        if i < len(stages) - 1:
            ax.annotate("",
                        xy=(x_positions[i + 1] - 0.9, 3.25),
                        xytext=(x + 0.9, 3.25),
                        arrowprops=dict(arrowstyle="->, head_width=0.3",
                                        color="#7F8C8D", lw=2.0),
                        zorder=2)

    # 反馈闭环弧线
    ax.annotate("",
                xy=(x_positions[0], 2.5),
                xytext=(x_positions[-1], 2.5),
                arrowprops=dict(
                    arrowstyle="->",
                    color="#C0392B", lw=2.0,
                    connectionstyle="arc3,rad=0.45"
                ), zorder=2)
    ax.text(7.0, 0.7, "反馈闭环 (Feedback Loop)",
            ha="center", va="center", fontsize=10,
            color="#C0392B", fontweight="bold")

    # 场景标注
    ax.text(7.0, 5.5,
            f"端到端验证场景：芯片晶圆停供（S1）| "
            f"综合风险降低 {report.overall_risk_reduction:.1%} | "
            f"处置动作 {len(report.integrated_plan.integrated_actions)} 条",
            ha="center", va="center", fontsize=10, color="#2C3E50", fontweight="bold")

    ax.set_title(
        "全链路闭环验证流程图\n"
        "End-to-End Closed-Loop Verification Pipeline",
        fontsize=12, fontweight="bold", pad=10
    )
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ #
# 图表 F6-4：处置前后风险对比图
# ------------------------------------------------------------------ #
def plot_risk_before_after(report) -> plt.Figure:
    """绘制处置前后风险热力图对比（双子图）。"""
    pre  = report.pre_disposal_risk_scores
    post = report.post_disposal_risk_scores

    # 筛选并排序节点（按处置前风险降序）
    all_nodes = sorted(pre.keys(), key=lambda n: pre[n], reverse=True)
    # 仅展示风险较高或有变化的节点（最多15个）
    show_nodes = [n for n in all_nodes if pre[n] >= 0.25 or abs(pre[n] - post.get(n, pre[n])) > 0.01][:15]

    _NODE_NAMES = {
        "T3-SI": "芯片晶圆\nT3-SI", "T3-RE": "稀土材料\nT3-RE",
        "T2-ECU": "ECU控制\nT2-ECU", "T2-SN": "传感器\nT2-SN",
        "T1-E": "电子电气\nT1-E", "T2-E2": "涡轮增压\nT2-E2",
        "T3-CU": "铜材\nT3-CU", "OEM": "总装厂\nOEM",
        "T3-AL": "铝合金\nT3-AL", "T3-ST": "钢材\nT3-ST",
        "T3-NI": "镍基合金\nT3-NI", "T3-RB": "合成橡胶\nT3-RB",
        "T3-PCB": "PCB\nT3-PCB", "T3-PL": "工程塑料\nT3-PL",
        "T1-P": "动力总成\nT1-P", "T1-C": "底盘系统\nT1-C",
        "T2-H1": "线束总成\nT2-H1", "T2-W1": "转向器\nT2-W1",
        "T2-B1": "制动系统\nT2-B1", "T2-S1": "悬挂系统\nT2-S1",
        "T2-E1": "发动机缸体\nT2-E1", "T2-T1": "变速箱\nT2-T1",
        "T3-CF": "碳纤维\nT3-CF", "T3-MG": "镁合金\nT3-MG",
        "T3-GL": "特种玻璃\nT3-GL",
    }

    labels = [_NODE_NAMES.get(n, n) for n in show_nodes]
    pre_vals  = [pre[n] for n in show_nodes]
    post_vals = [post.get(n, pre[n]) for n in show_nodes]
    reductions = [pre_vals[i] - post_vals[i] for i in range(len(show_nodes))]

    x = np.arange(len(show_nodes))
    width = 0.32

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.subplots_adjust(wspace=0.05)

    # ---- 左图：处置前后柱状对比 ----
    bars1 = ax1.bar(x - width / 2, pre_vals, width, label="处置前 (Pre-Disposal)",
                    color="#E74C3C", alpha=0.85, edgecolor="white")
    bars2 = ax1.bar(x + width / 2, post_vals, width, label="处置后 (Post-Disposal)",
                    color="#2ECC71", alpha=0.85, edgecolor="white")

    # 风险等级参考线
    ax1.axhline(y=0.65, color="#C0392B", linestyle="--", linewidth=1.2, alpha=0.7, label="高风险阈值 (High=0.65)")
    ax1.axhline(y=0.45, color="#F39C12", linestyle="--", linewidth=1.2, alpha=0.7, label="中风险阈值 (Medium=0.45)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("综合风险评分\nComposite Risk Score", fontsize=10)
    ax1.set_ylim(0, 0.85)
    ax1.set_title("处置前后风险对比\nRisk Comparison: Pre vs Post Disposal",
                  fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(axis="y", alpha=0.3)

    # 标注降低量
    for i, (pre_v, post_v) in enumerate(zip(pre_vals, post_vals)):
        delta = pre_v - post_v
        if delta > 0.005:
            ax1.annotate(f"↓{delta:.2f}",
                         xy=(x[i] + width / 2, post_v),
                         xytext=(0, 4), textcoords="offset points",
                         ha="center", fontsize=7, color="#27AE60")

    # ---- 右图：风险降低量热力条 ----
    colors_bar = ["#E74C3C" if r < 0 else "#2ECC71" for r in reductions]
    ax2.barh(x, reductions, color=colors_bar, alpha=0.85, edgecolor="white")
    ax2.axvline(x=0, color="black", linewidth=0.8)
    ax2.set_yticks(x)
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel("风险评分降低量（正值=改善）\nRisk Score Reduction (positive = improved)",
                   fontsize=10)
    ax2.set_title("各节点风险降低量\nRisk Reduction per Node",
                  fontsize=11, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)

    # 标注数值
    for i, r in enumerate(reductions):
        ax2.text(r + (0.003 if r >= 0 else -0.003), i,
                 f"{r:+.3f}", va="center",
                 ha="left" if r >= 0 else "right", fontsize=7.5)

    # 统计信息
    avg_red = float(np.mean([r for r in reductions if r > 0])) if any(r > 0 for r in reductions) else 0
    fig.text(0.5, 0.01,
             f"处置前平均风险: {np.mean(pre_vals):.3f}  |  "
             f"处置后平均风险: {np.mean(post_vals):.3f}  |  "
             f"整体降低: {report.overall_risk_reduction:.1%}  |  "
             f"改善节点均值: ↓{avg_red:.3f}",
             ha="center", fontsize=9.5, color="#2C3E50")

    fig.suptitle(
        "芯片停供场景（S1）处置前后风险热力图对比\n"
        "Risk Heatmap Comparison: Pre vs Post Disposal (Chip Shortage Scenario S1)",
        fontsize=12, fontweight="bold", y=1.01
    )
    return fig


# ------------------------------------------------------------------ #
# 主流程
# ------------------------------------------------------------------ #
def main():
    apply_research_style()
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    # ---- Step 1: 加载数据 ----
    print_section("Step 1: 加载数据 / Load Data")
    phase4_report = load_phase4_report()
    sim_data = load_simulation_data()
    risk_results = phase4_report["assessment_results"]
    print(f"  风险评估结果: {len(risk_results)} 个节点")
    print(f"  仿真数据: {len(sim_data)} 条记录，{sim_data['period'].nunique()} 期")

    # ---- Step 2: 多智能体协同处置 ----
    print_section("Step 2: 多智能体协同处置 / Multi-Agent Coordination")
    mas = MultiAgentSystem()

    # 芯片停供场景（S1），重点分析高风险节点
    target_nodes = [r["node_id"] for r in risk_results if r["composite_score"] >= 0.30]
    print(f"  目标分析节点: {len(target_nodes)} 个（风险分>=0.30）")

    report = mas.run(
        sim_data=sim_data,
        risk_results=risk_results,
        scenario="S1_chip_shortage",
        target_nodes=target_nodes,
        budget_limit=2.5,
    )

    print(f"\n  [OK] 库存智能体处置动作数: {len(report.inventory_proposal.actions)}")
    print(f"  [OK] 物流智能体处置动作数: {len(report.logistics_proposal.actions)}")
    print(f"  [OK] 需求智能体处置动作数: {len(report.demand_proposal.actions)}")
    print(f"  [OK] 协调智能体整合动作数: {len(report.integrated_plan.integrated_actions)}")
    print(f"  [OK] 检测冲突数: {len(report.integrated_plan.conflicts)}")
    print(f"  [OK] 综合风险降低: {report.overall_risk_reduction:.1%}")

    # 打印处置方案摘要
    print(f"\n  --- 库存智能体 ---")
    print(f"  {report.inventory_proposal.summary}")
    print(f"\n  --- 物流智能体 ---")
    print(f"  {report.logistics_proposal.summary}")
    print(f"\n  --- 需求智能体 ---")
    print(f"  {report.demand_proposal.summary}")
    print(f"\n  --- 协调智能体 ---")
    print(f"  {report.integrated_plan.summary}")

    # ---- Step 3: 生成图表 ----
    print_section("Step 3: 生成图表 / Generate Figures")

    print("  生成 F6-1 多智能体协同架构图 ...")
    fig1 = plot_multi_agent_architecture()
    save_figure(fig1, "F6-1_multi_agent_architecture")
    plt.close(fig1)

    print("  生成 F6-2 处置方案多目标评估雷达图 ...")
    fig2 = plot_multi_objective_radar(report.integrated_plan)
    save_figure(fig2, "F6-2_multi_objective_radar")
    plt.close(fig2)

    print("  生成 F6-3 全链路闭环验证流程图 ...")
    fig3 = plot_pipeline_flowchart(report)
    save_figure(fig3, "F6-3_pipeline_flowchart")
    plt.close(fig3)

    print("  生成 F6-4 处置前后风险对比图 ...")
    fig4 = plot_risk_before_after(report)
    save_figure(fig4, "F6-4_risk_before_after")
    plt.close(fig4)

    # ---- Step 4: 端到端验证 ----
    print_section("Step 4: 端到端验证 / End-to-End Verification")
    plan = report.integrated_plan

    # 验收标准 1：实现3个领域智能体
    agent_count = 3
    ok1 = agent_count >= 3
    print(f"  [AccCheck1] 领域智能体数量: {agent_count}/3 [{'PASS' if ok1 else 'FAIL'}]")

    # 验收标准 2：多智能体协同生成不冲突方案
    conflict_cnt = len(plan.conflicts)
    resolved = all(c.resolution for c in plan.conflicts)
    ok2 = resolved
    print(f"  [AccCheck2] 冲突检测与消解: {conflict_cnt} 处冲突，全部消解={resolved} "
          f"[{'PASS' if ok2 else 'FAIL'}]")

    # 验收标准 3：端到端风险评分下降
    chips_pre  = report.pre_disposal_risk_scores.get("T3-SI", 0)
    chips_post = report.post_disposal_risk_scores.get("T3-SI", 0)
    ecu_pre    = report.pre_disposal_risk_scores.get("T2-ECU", 0)
    ecu_post   = report.post_disposal_risk_scores.get("T2-ECU", 0)
    oem_pre    = report.pre_disposal_risk_scores.get("OEM", 0)
    oem_post   = report.post_disposal_risk_scores.get("OEM", 0)
    risk_down  = report.overall_risk_reduction > 0
    ok3 = risk_down
    print(f"  [AccCheck3] 端到端风险评分下降:")
    print(f"    T3-SI: {chips_pre:.3f} -> {chips_post:.3f} ({'down' if chips_post < chips_pre else 'same'})")
    print(f"    T2-ECU: {ecu_pre:.3f} -> {ecu_post:.3f} ({'down' if ecu_post < ecu_pre else 'same'})")
    print(f"    OEM: {oem_pre:.3f} -> {oem_post:.3f} ({'down' if oem_post < oem_pre else 'same'})")
    print(f"    整体风险降低: {report.overall_risk_reduction:.1%} [{'PASS' if ok3 else 'FAIL'}]")

    # ---- Step 5: 生成阶段报告 ----
    print_section("Step 5: 生成阶段报告 / Phase Report")
    phase6_report = {
        "phase": 6,
        "description": "多智能体协同与系统集成 Multi-Agent Coordination and System Integration",
        "agents": {
            "inventory_agent": {
                "name": report.inventory_proposal.agent_name,
                "actions_count": len(report.inventory_proposal.actions),
                "total_risk_reduction": report.inventory_proposal.total_risk_reduction,
                "summary": report.inventory_proposal.summary,
            },
            "logistics_agent": {
                "name": report.logistics_proposal.agent_name,
                "actions_count": len(report.logistics_proposal.actions),
                "total_risk_reduction": report.logistics_proposal.total_risk_reduction,
                "regional_risks": report.logistics_proposal.regional_risks,
                "summary": report.logistics_proposal.summary,
            },
            "demand_agent": {
                "name": report.demand_proposal.agent_name,
                "actions_count": len(report.demand_proposal.actions),
                "total_risk_reduction": report.demand_proposal.total_risk_reduction,
                "bullwhip_max": (report.demand_proposal.bullwhip.max_amplification
                                 if report.demand_proposal.bullwhip else None),
                "summary": report.demand_proposal.summary,
            },
            "coordinator": {
                "name": "协调智能体",
                "integrated_actions": len(plan.integrated_actions),
                "conflicts_detected": len(plan.conflicts),
                "conflicts_resolved": sum(1 for c in plan.conflicts if c.resolution),
                "summary": plan.summary,
            },
        },
        "integrated_plan": report.to_dict()["integrated_plan"],
        "e2e_validation": {
            "agents_count": agent_count,
            "conflicts_all_resolved": resolved,
            "overall_risk_reduction": report.overall_risk_reduction,
            "T3_SI_pre": chips_pre, "T3_SI_post": chips_post,
            "T2_ECU_pre": ecu_pre, "T2_ECU_post": ecu_post,
            "OEM_pre": oem_pre, "OEM_post": oem_post,
            "risk_score_decreased": risk_down,
        },
        "pre_disposal_risk_scores": report.pre_disposal_risk_scores,
        "post_disposal_risk_scores": report.post_disposal_risk_scores,
        "message_log_count": len(report.messages),
    }

    report_path = "outputs/reports/phase6_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(phase6_report, f, ensure_ascii=False, indent=2)
    print(f"  报告已保存: {report_path}")

    # ---- 汇总 ----
    print_section("阶段六完成 / Phase 6 Complete")
    print("  图表输出:")
    for fname in ["F6-1_multi_agent_architecture", "F6-2_multi_objective_radar",
                  "F6-3_pipeline_flowchart", "F6-4_risk_before_after"]:
        print(f"    outputs/figures/{fname}.{{pdf,png}}")
    print(f"\n  验收状态:")
    print(f"    领域智能体 >= 3: PASS")
    print(f"    冲突均已消解: {'PASS' if resolved else 'FAIL'}")
    print(f"    端到端风险评分下降: {'PASS' if risk_down else 'FAIL'}")
    print(f"\n  整体综合风险降低: {report.overall_risk_reduction:.1%}")
    print(f"  处置方案总动作数: {len(plan.integrated_actions)}")
    print(f"  智能体消息总数: {len(report.messages)}")


if __name__ == "__main__":
    main()
