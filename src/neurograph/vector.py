"""Baseline retrieval over abstracts: BM25 (default) or TF-IDF cosine, pure Python, no model download."""
from __future__ import annotations

import math
import re
from collections import Counter

import networkx as nx

from . import schema as S

_STOP = frozenset("""a an and are as at be by for from in is it of on or that the to was were which with this
these those we our their its into than then also can has have had not but using used use based""".split())
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-']*")


def tokenize(text: str) -> list[str]:
    return [t for t in (w.lower().strip("-'") for w in _TOKEN.findall(text)) if t and t not in _STOP]


class Retriever:
    def __init__(self, docs: dict[str, str], kind: str = "bm25", k1: float = 1.5, b: float = 0.75,
                 titles: dict[str, str] | None = None):
        if kind not in ("bm25", "tfidf"):
            raise ValueError("kind must be 'bm25' or 'tfidf'")
        self.kind, self.k1, self.b = kind, k1, b
        self.docs, self.titles = docs, titles or {}
        self.ids = list(docs)
        self.tf = [Counter(tokenize(docs[i])) for i in self.ids]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = (sum(self.len) / len(self.len)) if self.len else 0.0
        df = Counter(t for c in self.tf for t in c)
        n = len(self.ids)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        if kind == "tfidf":
            self.norm = [math.sqrt(sum(((1 + math.log(f)) * self.idf[t]) ** 2 for t, f in c.items()))
                         for c in self.tf]

    @classmethod
    def from_graph(cls, g: nx.DiGraph, kind: str = "bm25") -> "Retriever":
        docs = {a["name"]: f"{a.get('title', '')}. {a.get('abstract', '')}"
                for _, a in g.nodes(data=True) if a["type"] == S.PAPER}
        titles = {a["name"]: a.get("title", "") for _, a in g.nodes(data=True) if a["type"] == S.PAPER}
        return cls(docs, kind, titles=titles)

    def search(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        q = Counter(tokenize(query))
        scores = []
        for i, c in enumerate(self.tf):
            s = 0.0
            if self.kind == "bm25":
                for t in q:
                    f = c.get(t)
                    if f:
                        s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            else:
                qn = math.sqrt(sum(((1 + math.log(f)) * self.idf.get(t, 0)) ** 2 for t, f in q.items())) or 1.0
                dot = sum((1 + math.log(f)) * self.idf.get(t, 0) * (1 + math.log(c[t])) * self.idf[t]
                          for t, f in q.items() if t in c)
                s = dot / (qn * self.norm[i]) if self.norm[i] else 0.0
            if s > 0:
                scores.append((self.ids[i], s))
        scores.sort(key=lambda x: (-x[1], x[0]))
        return scores[:k]
