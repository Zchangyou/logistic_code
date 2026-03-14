"""
供应链知识库模块
Supply Chain Knowledge Base Module

功能：
- 加载历史风险事件与处置策略知识
- 基于TF-IDF的文本检索（主路径）
- 支持FAISS向量检索（可选增强）
- 知识增量更新接口
"""
import json
import os
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class KnowledgeDocument:
    """知识文档"""
    id: str
    title: str
    title_en: str
    category: str
    risk_type: str
    content: str          # 拼接后的全文，用于检索
    raw: Dict             # 原始JSON数据
    keywords: List[str] = field(default_factory=list)


class SupplyChainKnowledgeBase:
    """供应链知识库

    支持两类知识：
    1. 历史风险事件（risk_events.json）
    2. 处置策略（strategies.json）
    """

    def __init__(self, knowledge_dir: Optional[str] = None):
        if knowledge_dir is None:
            base = Path(__file__).parent.parent.parent
            knowledge_dir = str(base / "data" / "knowledge")

        self.knowledge_dir = Path(knowledge_dir)
        self.risk_events: List[KnowledgeDocument] = []
        self.strategies: List[Dict] = []

        # TF-IDF检索器（懒加载）
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None
        self._doc_ids: List[str] = []

        self._load()
        self._build_index()

    # ------------------------------------------------------------------ #
    # 数据加载
    # ------------------------------------------------------------------ #
    def _load(self):
        events_path = self.knowledge_dir / "risk_events.json"
        if events_path.exists():
            with open(events_path, encoding="utf-8") as f:
                raw_events = json.load(f)
            for ev in raw_events:
                content = self._build_content(ev)
                doc = KnowledgeDocument(
                    id=ev["id"],
                    title=ev["title"],
                    title_en=ev["title_en"],
                    category=ev["category"],
                    risk_type=ev["risk_type"],
                    content=content,
                    raw=ev,
                    keywords=ev.get("keywords", []),
                )
                self.risk_events.append(doc)

        strategies_path = self.knowledge_dir / "strategies.json"
        if strategies_path.exists():
            with open(strategies_path, encoding="utf-8") as f:
                self.strategies = json.load(f)

    @staticmethod
    def _build_content(ev: Dict) -> str:
        """将事件字典拼接为检索用文本"""
        parts = [
            ev.get("title", ""),
            ev.get("title_en", ""),
            ev.get("risk_type", ""),
            ev.get("trigger", ""),
            ev.get("propagation_path", ""),
            " ".join(ev.get("affected_nodes", [])),
            " ".join(ev.get("keywords", [])),
            " ".join(ev.get("resolution_measures", [])),
        ]
        return " ".join(p for p in parts if p)

    # ------------------------------------------------------------------ #
    # TF-IDF索引构建
    # ------------------------------------------------------------------ #
    def _build_index(self):
        if not self.risk_events:
            return
        corpus = [doc.content for doc in self.risk_events]
        self._doc_ids = [doc.id for doc in self.risk_events]

        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",   # 字符级n-gram，适合中文
            ngram_range=(1, 3),
            max_features=5000,
            sublinear_tf=True,
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(corpus)

    # ------------------------------------------------------------------ #
    # 检索接口
    # ------------------------------------------------------------------ #
    def search(self, query: str, top_k: int = 3) -> List[KnowledgeDocument]:
        """基于TF-IDF检索相关知识文档"""
        if self._vectorizer is None or self._tfidf_matrix is None:
            return []

        q_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self._tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(self.risk_events[idx])
        return results

    def get_strategies_for_risk(
        self, risk_type: str, node_data: Optional[Dict] = None, top_k: int = 3
    ) -> List[Dict]:
        """根据风险类型和节点数据筛选并排序策略"""
        candidates = [
            s for s in self.strategies
            if risk_type in s.get("applicable_risks", [])
        ]

        if not candidates:
            candidates = self.strategies  # 兜底：返回全部

        # 根据节点数据调整可行性分
        scored = []
        for s in candidates:
            score = s["feasibility_base"] * s["risk_reduction"]
            if node_data:
                conds = s.get("priority_conditions", {})
                for metric, threshold in conds.items():
                    val = node_data.get(metric)
                    if val is not None:
                        if metric in ("capacity_utilization", "demand_volatility"):
                            if val >= threshold:
                                score *= 1.2   # 符合触发条件，优先级提升
                        elif metric in ("inventory_level", "on_time_delivery",
                                        "logistics_reliability"):
                            if val <= threshold:
                                score *= 1.2
                        elif metric == "supplier_count":
                            if val <= threshold:
                                score *= 1.3
            scored.append((score, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:top_k]]

    def add_event(self, event: Dict):
        """增量添加知识事件"""
        content = self._build_content(event)
        doc = KnowledgeDocument(
            id=event["id"],
            title=event["title"],
            title_en=event.get("title_en", ""),
            category=event["category"],
            risk_type=event["risk_type"],
            content=content,
            raw=event,
            keywords=event.get("keywords", []),
        )
        self.risk_events.append(doc)
        # 重建索引
        self._build_index()

    # ------------------------------------------------------------------ #
    # 统计信息
    # ------------------------------------------------------------------ #
    def summary(self) -> Dict:
        return {
            "total_events": len(self.risk_events),
            "total_strategies": len(self.strategies),
            "categories": list({doc.category for doc in self.risk_events}),
            "risk_types": list({doc.risk_type for doc in self.risk_events}),
        }
