"""Question answering: GraphRAG (entity linking -> traversal -> subgraph context -> answer) and a vector baseline.

Both modes share the same question parser (entity linking + intent), so the comparison isolates what the
graph adds: multi-hop traversal and set logic, versus reading the top-k retrieved abstracts.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from itertools import islice

import networkx as nx

from . import schema as S
from .extract_offline import extract_text
from .ontology import Mention, family_of, find_mentions
from .schema import node_id
from .vector import Retriever

_TARGET_PATTERNS = [
    (S.PAPER, re.compile(r"\b(papers?|studies|study|articles?|works?)\b")),
    (S.METHOD, re.compile(r"\b(methods?|classifiers?|algorithms?|approaches|techniques?|models?)\b")),
    (S.DATASET, re.compile(r"\bdatasets?\b")),
    (S.PARADIGM, re.compile(r"\bparadigms?\b")),
    (S.FAMILY, re.compile(r"\bfamil(?:y|ies)\b")),
]
_GROUP = re.compile(r"\b(?:per|by|for each|each)\s+(method|dataset|paradigm|famil(?:y|ies))")
_GROUP_TYPE = {"method": S.METHOD, "dataset": S.DATASET, "paradigm": S.PARADIGM, "famil": S.FAMILY}


def _group_type(word: str) -> str:
    return _GROUP_TYPE["famil" if word.startswith("famil") else word]
_COUNT = re.compile(r"\bhow many\b|\bcount\b|\bnumber of\b")
_ARGMAX = re.compile(r"\b(highest|best|top|maximum|greatest|largest)\b")
_PATH = re.compile(r"\b(connect\w*|relat\w*|link\w*|path|between|bridge\w*)\b")
_BOTH = re.compile(r"\bboth\b")
_SLOT_TYPES = (S.METHOD, S.FAMILY, S.DATASET, S.PARADIGM)


@dataclass
class Parsed:
    question: str
    mentions: list[Mention]
    target: str = S.PAPER
    op: str = "list"  # list | count | count_by | argmax | path
    both: bool = False
    group_by: str | None = None
    metric: str = "accuracy"

    @property
    def constraints(self) -> list[Mention]:
        return [m for m in self.mentions if m.type in _SLOT_TYPES]


@dataclass
class Answer:
    mode: str
    question: str
    parsed: Parsed | None
    items: list[str] = field(default_factory=list)       # paper ids / entity names
    counts: dict[str, int] | None = None                  # for count questions
    evidence: list[str] = field(default_factory=list)     # paper ids supporting the answer
    paths: list[list[str]] = field(default_factory=list)  # node-id paths (path questions)
    text: str = ""
    context: str = ""                                     # assembled context (LLM prompt / audit)


def parse_question(question: str) -> Parsed:
    mentions = find_mentions(question)
    masked = list(question.lower())
    for m in mentions:
        for i in range(m.start, m.end):
            masked[i] = " "
    low = "".join(masked)
    p = Parsed(question, mentions, both=bool(_BOTH.search(low)))

    hits = [(mm.start(), t) for t, rx in _TARGET_PATTERNS if (mm := rx.search(low))]
    if hits:
        p.target = min(hits)[1]
    for m in mentions:
        if m.type == S.METRIC:
            p.metric = m.name
    non_metric = [m for m in mentions if m.type != S.METRIC]
    g = _GROUP.search(low)
    if len({(m.type, m.name) for m in non_metric}) >= 2 and _PATH.search(low):
        p.op = "path"
    elif _COUNT.search(low):
        p.op = "count_by" if g else "count"
        if g:
            p.group_by = _group_type(g.group(1))
    elif _ARGMAX.search(low):
        p.op = "argmax"
    return p


# ------------------------------------------------------------------ graph helpers

def _pid(paper_id: str) -> str:
    return node_id(S.PAPER, paper_id)


def _strip(nid: str) -> str:
    return nid.split(":", 1)[1]


def _preds(g, nid, rel) -> set[str]:
    return {u for u, _, a in g.in_edges(nid, data=True) if a["rel"] == rel}


def _succs(g, nid, rel) -> set[str]:
    return {v for _, v, a in g.out_edges(nid, data=True) if a["rel"] == rel}


_REL_FOR = {S.METHOD: S.USES_METHOD, S.DATASET: S.EVALUATED_ON, S.PARADIGM: S.STUDIES, S.METRIC: S.REPORTS}


def papers_of(g: nx.DiGraph, m: Mention) -> set[str]:
    """1-hop (or 2-hop for method families) reverse traversal from an entity to its papers."""
    nid = node_id(m.type, m.name)
    if nid not in g:
        return set()
    if m.type == S.FAMILY:
        out: set[str] = set()
        for meth in _preds(g, nid, S.IS_A):
            out |= _preds(g, meth, S.USES_METHOD)
        return out
    return _preds(g, nid, _REL_FOR[m.type])


def select_papers(g: nx.DiGraph, mentions: list[Mention], both: bool) -> set[str]:
    slots: dict[str, list[set[str]]] = {}
    for m in mentions:
        if m.type in _SLOT_TYPES:
            slots.setdefault(m.type, []).append(papers_of(g, m))
    result: set[str] | None = None
    for sets in slots.values():
        combined = set.intersection(*sets) if both else set.union(*sets)
        result = combined if result is None else result & combined
    return result if result is not None else set()


def entities_of(g: nx.DiGraph, papers: set[str], target: str) -> Counter:
    """Count, for each entity of `target` type, the papers (from `papers`) linked to it."""
    c: Counter = Counter()
    for p in papers:
        if target == S.FAMILY:
            ents = {f for m in _succs(g, p, S.USES_METHOD) for f in _succs(g, m, S.IS_A)}
        else:
            ents = _succs(g, p, _REL_FOR[target])
        for e in ents:
            c[e] += 1
    return c


def _label(g, nid) -> str:
    return g.nodes[nid]["name"]


def paper_facts(g: nx.DiGraph, pid: str) -> str:
    a = g.nodes[pid]
    parts = []
    for rel in (S.USES_METHOD, S.EVALUATED_ON, S.STUDIES):
        names = sorted(_label(g, v) for v in _succs(g, pid, rel))
        if names:
            parts.append(f"{rel}: {', '.join(names)}")
    rep = [f"{_label(g, v)}={d['value']:g}{'%' if d.get('unit') == '%' else ''}"
           for _, v, d in g.out_edges(pid, data=True) if d["rel"] == S.REPORTS]
    if rep:
        parts.append("REPORTS: " + ", ".join(sorted(rep)))
    extra = []
    if a.get("channels"):
        extra.append(f"{a['channels']} channels")
    if a.get("sampling_rate_hz"):
        extra.append(f"{a['sampling_rate_hz']:g} Hz")
    if extra:
        parts.append("RECORDING: " + ", ".join(extra))
    return f"[{a['name']}] \"{a.get('title', '')}\" | " + " | ".join(parts)


def _describe(mentions: list[Mention], both: bool) -> str:
    c = [m for m in mentions if m.type in _SLOT_TYPES]
    return (" AND " if both else " + ").join(f"{m.name}" for m in c) or "(no entities)"


def _title(g, pid) -> str:
    return g.nodes[pid].get("title", "")


# ------------------------------------------------------------------ graph mode

def graph_answer(g: nx.DiGraph, question: str) -> Answer:
    p = parse_question(question)
    ans = Answer("graph", question, p)
    cons = p.constraints
    if not cons:
        ans.text = "GraphRAG could not link any known entity (method, dataset, paradigm) in the question."
        return ans
    if p.op == "path":
        return _graph_path(g, p, ans)
    if p.op == "count_by":
        return _graph_count_by(g, p, ans)
    if p.op == "argmax":
        return _graph_argmax(g, p, ans)

    papers, common = _select_with_both(g, p)
    if common is not None:  # keep only papers that link to a bridging entity
        papers = {x for x in papers if {_strip(e) for e in entities_of(g, {x}, p.target)} & common}
    ans.evidence = sorted(_strip(x) for x in papers)
    if p.op == "count":
        ans.counts = {"papers": len(papers)}
        ans.items = ans.evidence
        ans.text = f"{len(papers)} paper(s) match {_describe(cons, p.both)}."
    elif p.target == S.PAPER:
        ans.items = ans.evidence
        ans.text = (f"Papers matching {_describe(cons, p.both)}: " +
                    ("; ".join(f"{i} ({_title(g, _pid(i))})" for i in ans.items) or "none") + ".")
    else:
        ans.items = sorted(_target_items(g, p, papers) & common if common is not None
                           else _target_items(g, p, papers))
        ans.text = (f"{p.target}s linked to {_describe(cons, p.both)}: {', '.join(ans.items) or 'none'}. "
                    f"Supporting papers: {', '.join(ans.evidence) or 'none'}.")
    ans.context = "\n".join(paper_facts(g, _pid(i)) for i in ans.evidence)
    return ans


def _exclusions(p: Parsed) -> set[str]:
    return {m.name for m in p.mentions if m.type == p.target}


def _family_filter(g, p: Parsed, nid: str) -> bool:
    fams = {m.name for m in p.mentions if m.type == S.FAMILY}
    return not fams or (p.target == S.METHOD and family_of(_strip(nid)) in fams)


def _target_items(g, p: Parsed, papers: set[str]) -> set[str]:
    names = set()
    for nid in entities_of(g, papers, p.target):
        if p.target == S.METHOD and not _family_filter(g, p, nid):
            continue
        names.add(_strip(nid))
    return names - _exclusions(p)


def _select_with_both(g, p: Parsed) -> tuple[set[str], set[str] | None]:
    """Select papers; for non-paper targets, 'both A and B' over a paradigm/dataset slot means the
    answer entities must be reachable via A *and* via B (returned as `common`)."""
    cons = p.constraints
    if p.both and p.target != S.PAPER:
        for t in (S.PARADIGM, S.DATASET):
            ms = [m for m in cons if m.type == t]
            if len(ms) > 1 and t != p.target:
                per_entity, all_papers = [], set()
                for m in ms:
                    sub = select_papers(g, [c for c in cons if c.type != t] + [m], False)
                    all_papers |= sub
                    per_entity.append(_target_items(g, p, sub))
                return all_papers, set.intersection(*per_entity)
    return select_papers(g, cons, p.both), None


def _graph_count_by(g, p: Parsed, ans: Answer) -> Answer:
    papers = select_papers(g, p.constraints, p.both)
    c = entities_of(g, papers, p.group_by)
    excl = {m.name for m in p.constraints if m.type == p.group_by}
    ans.counts = {_strip(k): v for k, v in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
                  if _strip(k) not in excl}
    ans.items = list(ans.counts)
    ans.evidence = sorted(_strip(x) for x in papers)
    ans.text = (f"Papers per {p.group_by.lower()} for {_describe(p.constraints, p.both)}: " +
                ", ".join(f"{k}={v}" for k, v in ans.counts.items()) + ".")
    ans.context = "\n".join(paper_facts(g, _pid(i)) for i in ans.evidence)
    return ans


def _graph_argmax(g, p: Parsed, ans: Answer) -> Answer:
    papers = select_papers(g, p.constraints, p.both)
    mid = node_id(S.METRIC, p.metric)
    scored = []
    for pid in papers:
        d = g.get_edge_data(pid, mid)
        if d and d.get("rel") == S.REPORTS:
            scored.append((d["value"], pid))
    ans.evidence = sorted(_strip(x) for x in papers)
    if not scored:
        ans.text = f"No paper matching {_describe(p.constraints, p.both)} reports {p.metric}."
        return ans
    best = max(v for v, _ in scored)
    winners = sorted(_strip(pid) for v, pid in scored if v == best)
    ans.items = winners
    ans.counts = None
    ans.text = (f"Highest {p.metric} among {len(scored)} paper(s) on {_describe(p.constraints, p.both)}: "
                f"{best:g} reported by " + "; ".join(f"{w} ({_title(g, _pid(w))})" for w in winners) + ".")
    ans.context = "\n".join(paper_facts(g, _pid(i)) for i in ans.evidence)
    return ans


def _graph_path(g, p: Parsed, ans: Answer) -> Answer:
    ends, seen = [], set()
    for m in p.constraints:
        key = (m.type, m.name)
        if key not in seen and node_id(*key) in g:
            seen.add(key)
            ends.append(node_id(*key))
    if len(ends) < 2:
        ans.text = "Could not find two entities in the graph to connect."
        return ans
    # Metric nodes (accuracy, ...) are generic hubs that connect everything; exclude them from path search.
    u = g.subgraph([n for n in g if g.nodes[n]["type"] != S.METRIC]).to_undirected()
    a, b = ends[0], ends[1]
    try:
        paths = list(islice(nx.all_shortest_paths(u, a, b), 20))
    except nx.NetworkXNoPath:
        ans.text = f"No path between {_strip(a)} and {_strip(b)}."
        return ans
    ans.paths = paths
    bridges = sorted({_strip(n) for path in paths for n in path[1:-1] if g.nodes[n]["type"] != S.PAPER})
    papers = sorted({_strip(n) for path in paths for n in path if g.nodes[n]["type"] == S.PAPER})
    ans.items, ans.evidence = bridges, papers
    hops = len(paths[0]) - 1
    ans.text = (f"{_strip(a)} and {_strip(b)} are {hops} hops apart. Example path: " +
                " -- ".join(_label(g, n) if g.nodes[n]["type"] != S.PAPER else _strip(n) for n in paths[0]) +
                f". Bridging entities: {', '.join(bridges) or 'none'}.")
    ans.context = "\n".join(paper_facts(g, _pid(i)) for i in papers)
    return ans


# ------------------------------------------------------------------ vector mode

def vector_answer(retriever: Retriever, question: str, k: int = 3, g: nx.DiGraph | None = None) -> Answer:
    """Baseline: retrieve top-k abstracts, then 'read' them with the dictionary extractor (no traversal)."""
    p = parse_question(question)
    hits = retriever.search(question, k)
    ids = [h[0] for h in hits]
    ans = Answer("vector", question, p, evidence=ids)
    docs = {i: retriever.docs[i] for i in ids}
    ex = {i: extract_text(t) for i, t in docs.items()}
    excl = _exclusions(p)

    def ents(i, target):
        e = ex[i]
        if target == S.METHOD:
            fams = {mm.name for mm in p.mentions if mm.type == S.FAMILY}
            return {m.name for m in e.methods if not fams or m.family in fams}
        if target == S.DATASET:
            return set(e.datasets)
        if target == S.PARADIGM:
            return set(e.paradigms)
        if target == S.FAMILY:
            return {m.family for m in e.methods if m.family}
        return set()

    if p.op == "count_by" and p.group_by:
        c = Counter(n for i in ids for n in ents(i, p.group_by) if n not in excl)
        ans.counts = dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))
        ans.items = list(ans.counts)
        ans.text = "Papers per entity among retrieved abstracts: " + ", ".join(f"{k}={v}" for k, v in ans.counts.items())
    elif p.op == "count":
        ans.counts = {"papers": len(ids)}
        ans.items = ids
        ans.text = f"{len(ids)} retrieved abstract(s) (a retriever cannot count the whole corpus)."
    elif p.op == "argmax":
        best = [(max((m.value for m in ex[i].metrics if m.name == p.metric), default=None), i) for i in ids]
        best = [(v, i) for v, i in best if v is not None]
        if best:
            top = max(v for v, _ in best)
            ans.items = sorted(i for v, i in best if v == top)
            ans.text = f"Highest {p.metric} among retrieved abstracts: {top:g} ({', '.join(ans.items)})."
    elif p.op == "path":
        names = set().union(*(ents(i, S.PARADIGM) for i in ids)) if ids else set()
        ans.items = sorted(names - {m.name for m in p.mentions})
        ans.text = "Paradigms co-occurring in retrieved abstracts: " + (", ".join(ans.items) or "none")
    elif p.target == S.PAPER:
        ans.items = ids
        ans.text = "Top retrieved abstracts: " + "; ".join(f"{i} ({retriever.titles.get(i, '')})" for i in ids)
    else:
        names = set().union(*(ents(i, p.target) for i in ids)) - excl if ids else set()
        ans.items = sorted(names)
        ans.text = f"{p.target}s mentioned in the top-{k} abstracts: {', '.join(ans.items) or 'none'}."
    ans.context = "\n".join(f"[{i}] {docs[i]}" for i in ids)
    return ans
