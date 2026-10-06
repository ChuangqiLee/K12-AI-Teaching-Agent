"""Knowledge-graph retrieval used for "problem comprehension" (Section 2.1).

The LLaMA model receives the retrieved node(s) as grounding context
("structured knowledge injection", Section 3.1) so that the generated
definition / derivation / application stays inside the curriculum.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ..schemas import Stage

_TOKEN = re.compile(r"[a-zA-Z]+|[一-鿿]|\d+|\S")


@dataclass
class KGNode:
    id: str
    stage: Stage
    data: dict

    def field(self, name: str, lang: str):
        value = self.data.get(name)
        if isinstance(value, dict) and lang in value:
            return value[lang]
        if isinstance(value, dict):
            return next(iter(value.values()))
        return value

    @property
    def prerequisites(self) -> List[str]:
        return list(self.data.get("prerequisites", []))


class KnowledgeBase:
    def __init__(self, nodes: List[KGNode]):
        self.nodes: Dict[str, KGNode] = {n.id: n for n in nodes}

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeBase":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        nodes = [KGNode(n["id"], Stage(n["stage"]), n) for n in raw["nodes"]]
        return cls(nodes)

    def get(self, node_id: str) -> Optional[KGNode]:
        return self.nodes.get(node_id)

    def score(self, node: KGNode, question: str, stage: Optional[Stage]) -> float:
        q = question.lower()
        score = 0.0
        for kw in node.data.get("keywords", []):
            if kw.lower() in q:
                score += 2.0 + 0.1 * len(kw)
        q_tokens = set(_TOKEN.findall(q))
        for lang in ("zh", "en"):
            topic = str(node.field("topic", lang)).lower()
            score += 0.2 * len(q_tokens & set(_TOKEN.findall(topic)))
        if stage is not None and node.stage == stage:
            score += 0.5
        return score

    def retrieve(self, question: str, stage: Optional[Stage] = None, top_k: int = 1,
                 with_prerequisites: bool = False) -> List[KGNode]:
        ranked = sorted(self.nodes.values(), key=lambda n: self.score(n, question, stage), reverse=True)
        hits = [n for n in ranked[:top_k] if self.score(n, question, stage) > 0.5]
        if with_prerequisites:
            seen = {n.id for n in hits}
            for n in list(hits):
                for pid in n.prerequisites:
                    p = self.get(pid)
                    if p is not None and pid not in seen:
                        hits.append(p)
                        seen.add(pid)
        return hits

    def extract_concepts(self, question: str, lang: str = "zh") -> List[str]:
        """Identify the essential concepts in a query (e.g. "right triangle", "hypotenuse")."""
        concepts: List[str] = []
        for node in self.retrieve(question, top_k=2):
            for c in node.field("concepts", lang) or []:
                if c not in concepts:
                    concepts.append(c)
        return concepts

    def context_block(self, nodes: List[KGNode], lang: str = "zh") -> str:
        """Serialise retrieved nodes into the prompt's knowledge section."""
        lines = []
        for n in nodes:
            lines.append(f"## {n.field('topic', lang)}")
            lines.append(f"- definition: {n.field('definition', lang)}")
            for i, step in enumerate(n.field("derivation", lang) or [], 1):
                lines.append(f"- step {i}: {step}")
            for f in n.data.get("formulas", []):
                lines.append(f"- formula: ${f}$")
            for a in n.field("applications", lang) or []:
                lines.append(f"- application: {a}")
        return "\n".join(lines)
