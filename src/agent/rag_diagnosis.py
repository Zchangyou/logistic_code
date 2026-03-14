"""
RAG 瓶颈智能诊断模块
RAG-based Bottleneck Diagnosis Module

流程：
1. 加载阶段四风险评估报告
2. 从知识库检索相关历史事件
3. 调用通义千问 API（dashscope）进行推理
4. 输出结构化诊断报告（含瓶颈定位、成因、影响范围、处置建议）

说明：
- 若 dashscope API 密钥未配置，自动切换到基于规则的本地诊断（Mock模式）
- Mock 模式输出格式与 LLM 模式完全一致，便于后续流程统一处理
"""
import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from src.agent.knowledge_base import SupplyChainKnowledgeBase, KnowledgeDocument


# ------------------------------------------------------------------ #
# 数据结构
# ------------------------------------------------------------------ #
@dataclass
class BottleneckInfo:
    node_id: str
    node_name: str
    risk_score: float
    risk_level: str
    bottleneck_type: str          # 材料短缺 / 供应商集中 / 物流脆弱 / 需求波动
    root_cause: str
    impact_scope: List[str]
    severity: str                 # 严重 / 中等 / 轻微


@dataclass
class DiagnosisReport:
    scenario: str
    primary_bottleneck: BottleneckInfo
    secondary_bottlenecks: List[BottleneckInfo]
    propagation_path: str
    related_events: List[str]      # 历史事件ID
    recommendations: List[str]
    confidence: float
    generated_by: str              # "llm" or "rule"
    raw_response: str = ""


# ------------------------------------------------------------------ #
# 主诊断类
# ------------------------------------------------------------------ #
class RAGDiagnosisEngine:
    """RAG 瓶颈诊断引擎"""

    # 节点中文名映射
    NODE_NAMES = {
        "T3-SI": "芯片晶圆供应商",
        "T3-RE": "稀土材料供应商",
        "T2-ECU": "ECU控制单元",
        "T2-SN": "传感器模组",
        "T1-E": "电子电气系统集成商",
        "T2-E2": "涡轮增压器",
        "T3-CU": "铜材供应商",
        "OEM": "总装厂",
        "T1-P": "动力总成系统集成商",
        "T1-C": "底盘系统集成商",
        "T3-ST": "特种钢材",
        "T3-AL": "铸造铝合金",
        "T3-NI": "镍基合金",
        "T3-RB": "合成橡胶",
        "T3-PCB": "印制电路板",
        "T3-PL": "工程塑料",
        "T3-CF": "碳纤维材料",
        "T3-MG": "镁合金",
        "T3-GL": "特种玻璃",
        "T2-E1": "发动机缸体",
        "T2-T1": "变速箱总成",
        "T2-B1": "制动系统",
        "T2-S1": "悬挂系统",
        "T2-W1": "转向器",
        "T2-H1": "线束总成",
    }

    # 风险类型映射（基于节点ID规则推断）
    RISK_TYPE_MAP = {
        "T3-SI": "材料短缺",
        "T3-RE": "供应商集中",
        "T2-ECU": "级联短缺",
        "T2-SN": "多重风险叠加",
        "T1-E": "上游风险汇聚",
        "T2-E2": "物流脆弱",
        "T3-CU": "潜在短缺",
        "OEM": "需求波动",
    }

    def __init__(self, knowledge_base: Optional[SupplyChainKnowledgeBase] = None):
        self.kb = knowledge_base or SupplyChainKnowledgeBase()
        self._llm_available = self._check_llm()

    def _check_llm(self) -> bool:
        """检查 dashscope API 是否可用（env var 优先，其次读 settings.yaml）"""
        api_key = os.environ.get("DASHSCOPE_API_KEY", "")

        if not api_key:
            # 从 settings.yaml 读取
            try:
                import yaml
                cfg_path = os.path.join(
                    os.path.dirname(__file__), "..", "..", "config", "settings.yaml"
                )
                with open(cfg_path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                api_key = cfg.get("llm", {}).get("api_key", "")
            except Exception:
                pass

        if not api_key:
            return False
        try:
            import dashscope
            dashscope.api_key = api_key
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------ #
    # 主入口
    # ------------------------------------------------------------------ #
    def diagnose(
        self,
        phase4_report: Dict,
        scenario: str = "芯片晶圆停供场景",
        top_bottlenecks: int = 3,
    ) -> DiagnosisReport:
        """执行诊断流程"""
        # 1. 提取高风险节点（场景感知：优先展示与场景相关的节点）
        top_nodes = self._extract_top_nodes(phase4_report, top_bottlenecks, scenario=scenario)

        # 2. 知识检索
        query = f"{scenario} {' '.join(n['node_id'] for n in top_nodes)}"
        relevant_docs = self.kb.search(query, top_k=3)

        # 3. 生成诊断报告（LLM 或 Rule-based）
        if self._llm_available:
            report = self._diagnose_with_llm(scenario, top_nodes, relevant_docs, phase4_report)
        else:
            report = self._diagnose_with_rules(scenario, top_nodes, relevant_docs, phase4_report)

        return report

    # ------------------------------------------------------------------ #
    # 节点提取
    # ------------------------------------------------------------------ #
    # 场景关键词 → 优先展示的节点ID
    SCENARIO_PRIORITY = {
        "芯片": "T3-SI",
        "chip": "T3-SI",
        "稀土": "T3-RE",
        "rare earth": "T3-RE",
        "物流": "T2-E2",
        "华东": "T2-E2",
        "需求": "OEM",
        "demand": "OEM",
    }

    def _extract_top_nodes(self, report: Dict, top_k: int, scenario: str = "") -> List[Dict]:
        results = report.get("assessment_results", [])
        sorted_nodes = sorted(results, key=lambda x: x["composite_score"], reverse=True)

        # 场景优先节点：如果场景中含有特定关键词，把对应节点提升为第一位
        priority_node_id = None
        for keyword, nid in self.SCENARIO_PRIORITY.items():
            if keyword in scenario:
                priority_node_id = nid
                break

        def sort_key(n):
            if priority_node_id and n["node_id"] == priority_node_id:
                return (0, -n["composite_score"])   # 置顶
            return (1, -n["composite_score"])

        sorted_nodes = sorted(results, key=sort_key)

        out = []
        for n in sorted_nodes[:top_k]:
            nid = n["node_id"]
            out.append({
                "node_id": nid,
                "node_name": self.NODE_NAMES.get(nid, nid),
                "composite_score": n["composite_score"],
                "risk_level": n["risk_level"],
                "risk_type": self.RISK_TYPE_MAP.get(nid, "综合风险"),
            })
        return out

    # ------------------------------------------------------------------ #
    # Rule-based 诊断（Mock 模式）
    # ------------------------------------------------------------------ #
    def _diagnose_with_rules(
        self,
        scenario: str,
        top_nodes: List[Dict],
        docs: List[KnowledgeDocument],
        report: Dict,
    ) -> DiagnosisReport:
        primary_node = top_nodes[0]
        secondary_nodes = top_nodes[1:]

        # 构建瓶颈信息
        primary_bottleneck = BottleneckInfo(
            node_id=primary_node["node_id"],
            node_name=primary_node["node_name"],
            risk_score=primary_node["composite_score"],
            risk_level=primary_node["risk_level"],
            bottleneck_type=primary_node["risk_type"],
            root_cause=self._infer_root_cause(primary_node["node_id"]),
            impact_scope=self._infer_impact_scope(primary_node["node_id"]),
            severity="严重" if primary_node["composite_score"] > 0.5 else "中等",
        )

        secondary_bottlenecks = [
            BottleneckInfo(
                node_id=n["node_id"],
                node_name=n["node_name"],
                risk_score=n["composite_score"],
                risk_level=n["risk_level"],
                bottleneck_type=n["risk_type"],
                root_cause=self._infer_root_cause(n["node_id"]),
                impact_scope=self._infer_impact_scope(n["node_id"]),
                severity="中等" if n["composite_score"] > 0.4 else "轻微",
            )
            for n in secondary_nodes
        ]

        # 传播路径
        propagation = self._infer_propagation_path(primary_node["node_id"])

        # 推荐措施（来自知识库中最相关事件）
        recommendations = []
        for doc in docs[:2]:
            measures = doc.raw.get("resolution_measures", [])[:2]
            recommendations.extend(measures)
        # 补充通用建议
        recommendations += [
            "建立实时供应链监控预警系统，对关键节点设置多级预警阈值",
            "制定业务连续性计划（BCP），明确各类风险场景的应急响应流程",
        ]

        related_event_ids = [doc.id for doc in docs]

        return DiagnosisReport(
            scenario=scenario,
            primary_bottleneck=primary_bottleneck,
            secondary_bottlenecks=secondary_bottlenecks,
            propagation_path=propagation,
            related_events=related_event_ids,
            recommendations=recommendations[:5],
            confidence=0.82,
            generated_by="rule",
            raw_response="[Rule-based diagnosis — LLM not available]",
        )

    def _infer_root_cause(self, node_id: str) -> str:
        causes = {
            "T3-SI": "全球半导体产能高度集中于台湾、韩国少数代工厂（台积电、三星），"
                     "供应商数量仅为1，产能利用率长期超过95%，无冗余产能应对需求波动；"
                     "叠加地缘政治风险与技术出口管制压力，形成高度脆弱的单点供应依赖。",
            "T3-RE": "全球稀土储量约60%集中于中国，包头基地为主要来源，"
                     "供应商数量=1，行业可替代来源极为有限；"
                     "产能利用率超过90%，在出口管制政策收紧时极易形成断供。",
            "T2-ECU": "ECU控制单元同时依赖芯片晶圆（T3-SI）和稀土材料（T3-RE）两个高风险源，"
                      "上游双重风险叠加；库存水位随供应收紧逐期下降，"
                      "交货准时率从期初0.95降至0.70，是典型的级联风险受害节点。",
            "T2-SN": "传感器模组依赖稀土（T3-RE）和芯片晶圆（T3-SI）两个核心原材料，"
                     "属于多重风险叠加节点；上游任一断供均会触发生产中断。",
            "T1-E": "电子电气系统集成商下属三个零部件供应商（ECU、线束、传感器）中，"
                    "ECU和传感器均受高风险上游影响，上游风险在此节点汇聚放大。",
            "T2-E2": "涡轮增压器位于无锡（华东地区），物流可靠性仅0.65，"
                     "跨区域长距离运输是其主要脆弱点，区域性灾害或物流中断时极易受影响。",
            "T3-CU": "铜材供应商数量仅2家，产能利用率超过85%，"
                     "全球铜价波动剧烈，供应链成本压力大，存在潜在短缺风险。",
            "OEM": "总装厂需求波动系数为0.35，受季节性市场需求影响明显，"
                   "订单波动向上游逐级放大（牛鞭效应），加剧全链路不稳定性。",
        }
        return causes.get(node_id, f"节点 {node_id} 存在综合性供应链风险，需进一步分析。")

    def _infer_impact_scope(self, node_id: str) -> List[str]:
        impacts = {
            "T3-SI": ["T2-ECU（ECU控制单元）", "T1-E（电子电气系统集成商）",
                      "OEM（总装厂）", "最终整车交付"],
            "T3-RE": ["T2-SN（传感器模组）", "T2-ECU（ECU控制单元）",
                      "T1-E（电子电气系统集成商）", "OEM（总装厂）"],
            "T2-ECU": ["T1-E（电子电气系统集成商）", "OEM（总装厂）"],
            "T2-SN": ["T1-E（电子电气系统集成商）", "OEM（总装厂）"],
            "T1-E": ["OEM（总装厂）", "最终整车交付"],
            "T2-E2": ["T1-P（动力总成系统集成商）", "OEM（总装厂）"],
            "OEM": ["整车产销计划", "下游经销商交付"],
        }
        return impacts.get(node_id, [self.NODE_NAMES.get(node_id, node_id)])

    def _infer_propagation_path(self, node_id: str) -> str:
        paths = {
            "T3-SI": "T3-SI（芯片晶圆停供）→ T2-ECU（库存耗尽，生产停滞）"
                     "→ T1-E（电气系统无法交付）→ OEM（总装线被迫减产）→ 最终交付延迟",
            "T3-RE": "T3-RE（稀土供应收紧）→ T2-SN, T2-ECU（生产受阻）"
                     "→ T1-E（系统集成缺件）→ OEM（整车减产）",
            "T2-ECU": "T2-ECU（级联短缺）→ T1-E → OEM",
        }
        return paths.get(
            node_id,
            f"{self.NODE_NAMES.get(node_id, node_id)} → 下游集成商 → OEM（总装厂）",
        )

    # ------------------------------------------------------------------ #
    # LLM 诊断模式
    # ------------------------------------------------------------------ #
    def _diagnose_with_llm(
        self,
        scenario: str,
        top_nodes: List[Dict],
        docs: List[KnowledgeDocument],
        report: Dict,
    ) -> DiagnosisReport:
        """调用通义千问 API 进行诊断"""
        try:
            import dashscope
            from dashscope import Generation

            context_parts = []
            for doc in docs:
                context_parts.append(
                    f"【历史事件：{doc.title}】\n"
                    f"触发原因：{doc.raw.get('trigger', '')}\n"
                    f"传播路径：{doc.raw.get('propagation_path', '')}\n"
                    f"处置措施：{'；'.join(doc.raw.get('resolution_measures', [])[:3])}"
                )
            context = "\n\n".join(context_parts)

            nodes_summary = "\n".join(
                f"  - {n['node_name']}（{n['node_id']}）：综合风险评分 {n['composite_score']:.3f}，"
                f"风险类型：{n['risk_type']}"
                for n in top_nodes
            )

            prompt = f"""你是一位供应链风险管理专家。请基于以下信息对汽车发动机供应链进行瓶颈诊断分析。

## 风险评估结果（按评分降序）
{nodes_summary}

## 相关历史事件参考
{context}

## 诊断场景
{scenario}

请按以下结构输出诊断报告（使用中文）：

**主要瓶颈节点**：[节点名称]
**瓶颈类型**：[材料短缺/供应商集中/物流脆弱/需求波动]
**根本原因分析**：[2-3句，说明风险成因]
**影响传播路径**：[从瓶颈节点到总装厂的传播链]
**影响范围**：[受影响的下游节点列表]
**严重程度**：[严重/中等/轻微]
**关联历史事件**：[最相关的1-2个历史事件名称]
**处置建议**：
1. [措施1]
2. [措施2]
3. [措施3]
**诊断置信度**：[0.0-1.0]
"""
            # 读取配置中的模型参数
            llm_model = "qwen-turbo"
            llm_max_tokens = 800
            llm_temperature = 0.3
            try:
                import yaml
                cfg_path = os.path.join(
                    os.path.dirname(__file__), "..", "..", "config", "settings.yaml"
                )
                with open(cfg_path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                llm_cfg = cfg.get("llm", {})
                llm_model = llm_cfg.get("model", llm_model)
                llm_max_tokens = llm_cfg.get("max_tokens", llm_max_tokens)
                llm_temperature = llm_cfg.get("temperature", llm_temperature)
            except Exception:
                pass

            resp = Generation.call(
                model=llm_model,
                prompt=prompt,
                max_tokens=llm_max_tokens,
                temperature=llm_temperature,
            )
            raw_text = resp.output.text if hasattr(resp, "output") else str(resp)

            # 解析 LLM 输出 → DiagnosisReport
            return self._parse_llm_output(raw_text, scenario, top_nodes, docs)

        except Exception as e:
            # LLM 调用失败，回退到规则诊断
            print(f"[WARNING] LLM 诊断失败（{e}），切换到规则诊断模式")
            return self._diagnose_with_rules(scenario, top_nodes, docs, report)

    def _parse_llm_output(
        self,
        text: str,
        scenario: str,
        top_nodes: List[Dict],
        docs: List[KnowledgeDocument],
    ) -> DiagnosisReport:
        """解析 LLM 输出文本为结构化报告"""
        import re

        def extract(pattern, default=""):
            m = re.search(pattern, text, re.DOTALL)
            return m.group(1).strip() if m else default

        primary_name = extract(r"\*\*主要瓶颈节点\*\*[：:]\s*(.+)")
        bottleneck_type = extract(r"\*\*瓶颈类型\*\*[：:]\s*(.+)")
        root_cause = extract(r"\*\*根本原因分析\*\*[：:]\s*(.+?)(?=\*\*|$)")
        propagation = extract(r"\*\*影响传播路径\*\*[：:]\s*(.+?)(?=\*\*|$)")
        severity = extract(r"\*\*严重程度\*\*[：:]\s*(.+)")
        confidence_str = extract(r"\*\*诊断置信度\*\*[：:]\s*(.+)")
        try:
            confidence = float(re.search(r"[\d.]+", confidence_str).group())
        except Exception:
            confidence = 0.85

        # 推荐措施
        recs = re.findall(r"\d+\.\s*(.+)", text)

        primary_node = top_nodes[0]
        primary_bottleneck = BottleneckInfo(
            node_id=primary_node["node_id"],
            node_name=primary_name or primary_node["node_name"],
            risk_score=primary_node["composite_score"],
            risk_level=primary_node["risk_level"],
            bottleneck_type=bottleneck_type or primary_node["risk_type"],
            root_cause=root_cause or self._infer_root_cause(primary_node["node_id"]),
            impact_scope=self._infer_impact_scope(primary_node["node_id"]),
            severity=severity or "中等",
        )

        secondary_bottlenecks = [
            BottleneckInfo(
                node_id=n["node_id"],
                node_name=n["node_name"],
                risk_score=n["composite_score"],
                risk_level=n["risk_level"],
                bottleneck_type=n["risk_type"],
                root_cause=self._infer_root_cause(n["node_id"]),
                impact_scope=self._infer_impact_scope(n["node_id"]),
                severity="中等",
            )
            for n in top_nodes[1:]
        ]

        return DiagnosisReport(
            scenario=scenario,
            primary_bottleneck=primary_bottleneck,
            secondary_bottlenecks=secondary_bottlenecks,
            propagation_path=propagation or self._infer_propagation_path(primary_node["node_id"]),
            related_events=[doc.id for doc in docs],
            recommendations=recs[:5] if recs else ["建议进一步分析供应链风险"],
            confidence=confidence,
            generated_by="llm",
            raw_response=text,
        )

    # ------------------------------------------------------------------ #
    # 序列化
    # ------------------------------------------------------------------ #
    def report_to_dict(self, report: DiagnosisReport) -> Dict:
        pb = report.primary_bottleneck
        return {
            "scenario": report.scenario,
            "generated_by": report.generated_by,
            "confidence": report.confidence,
            "primary_bottleneck": {
                "node_id": pb.node_id,
                "node_name": pb.node_name,
                "risk_score": pb.risk_score,
                "risk_level": pb.risk_level,
                "bottleneck_type": pb.bottleneck_type,
                "root_cause": pb.root_cause,
                "impact_scope": pb.impact_scope,
                "severity": pb.severity,
            },
            "secondary_bottlenecks": [
                {
                    "node_id": b.node_id,
                    "node_name": b.node_name,
                    "risk_score": b.risk_score,
                    "bottleneck_type": b.bottleneck_type,
                    "severity": b.severity,
                }
                for b in report.secondary_bottlenecks
            ],
            "propagation_path": report.propagation_path,
            "related_events": report.related_events,
            "recommendations": report.recommendations,
        }
