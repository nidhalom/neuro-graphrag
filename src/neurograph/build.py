"""Build the knowledge graph from papers with a pluggable extractor and store."""
from __future__ import annotations

from pathlib import Path

import networkx as nx

from . import schema as S
from .extract_offline import OfflineExtractor
from .fetch import load_jsonl
from .schema import Paper, PaperExtraction, node_id
from .store import GraphStore, NetworkxStore


def load_papers(path: str | Path) -> list[Paper]:
    return [Paper.model_validate(r) for r in load_jsonl(path)]


def add_paper(store: GraphStore, paper: Paper, ex: PaperExtraction) -> None:
    pid = node_id(S.PAPER, paper.id)
    store.upsert_node(pid, S.PAPER, paper.id, title=paper.title, abstract=paper.abstract,
                      authors=paper.authors, published=paper.published, url=paper.url,
                      source=paper.source, channels=ex.channels, sampling_rate_hz=ex.sampling_rate_hz)
    for m in ex.methods:
        mid = node_id(S.METHOD, m.name)
        store.upsert_node(mid, S.METHOD, m.name)
        store.upsert_edge(pid, mid, S.USES_METHOD)
        if m.family:
            fid = node_id(S.FAMILY, m.family)
            store.upsert_node(fid, S.FAMILY, m.family)
            store.upsert_edge(mid, fid, S.IS_A)
    for d in ex.datasets:
        did = node_id(S.DATASET, d)
        store.upsert_node(did, S.DATASET, d)
        store.upsert_edge(pid, did, S.EVALUATED_ON)
    for p in ex.paradigms:
        qid = node_id(S.PARADIGM, p)
        store.upsert_node(qid, S.PARADIGM, p)
        store.upsert_edge(pid, qid, S.STUDIES)
    for mv in ex.metrics:
        xid = node_id(S.METRIC, mv.name)
        store.upsert_node(xid, S.METRIC, mv.name)
        store.upsert_edge(pid, xid, S.REPORTS, value=mv.value, unit=mv.unit)


def build_graph(papers: list[Paper], extractor=None, store: GraphStore | None = None,
                on_error=None) -> GraphStore:
    extractor = extractor or OfflineExtractor()
    store = store or NetworkxStore()
    for paper in papers:
        try:
            ex = extractor.extract(paper)
        except Exception as e:  # one bad LLM response must not lose the whole build
            if on_error is None:
                raise
            on_error(paper, e)
            continue
        add_paper(store, paper, ex)
    return store


def graph_stats(g: nx.DiGraph) -> dict:
    by_type: dict[str, int] = {}
    for _, a in g.nodes(data=True):
        by_type[a["type"]] = by_type.get(a["type"], 0) + 1
    by_rel: dict[str, int] = {}
    for _, _, a in g.edges(data=True):
        by_rel[a["rel"]] = by_rel.get(a["rel"], 0) + 1
    return {"nodes": g.number_of_nodes(), "edges": g.number_of_edges(),
            "nodes_by_type": dict(sorted(by_type.items())), "edges_by_relation": dict(sorted(by_rel.items()))}
