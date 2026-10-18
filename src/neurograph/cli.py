"""Command line interface: neurograph fetch | build | ask | stats | eval | viz."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .build import build_graph, graph_stats, load_papers
from .extract_llm import LLMExtractor, llm_available
from .extract_offline import OfflineExtractor
from .store import NetworkxStore, get_store

DEFAULT_GRAPH = "out/graph.json"
DEFAULT_CORPUS = "data/fixture_abstracts.jsonl"


def _load(path: str):
    p = Path(path)
    if not p.exists():
        sys.exit(f"graph file not found: {p} (run `neurograph build` first)")
    return NetworkxStore.load_json(p).to_networkx()


def cmd_fetch(a):
    from .fetch import fetch
    papers = fetch(a.query or None, a.max_results, a.out, a.delay)
    print(f"wrote {len(papers)} abstracts to {a.out}")


def cmd_build(a):
    papers = load_papers(a.input)
    if a.extractor == "llm" or (a.extractor == "auto" and llm_available()):
        extractor = LLMExtractor()
    else:
        extractor = OfflineExtractor()
    print(f"extractor: {extractor.name}; papers: {len(papers)}")
    errors = []
    mem = build_graph(papers, extractor, NetworkxStore(), on_error=lambda p, e: errors.append((p.id, str(e))))
    for pid, e in errors:
        print(f"  skipped {pid}: {e}", file=sys.stderr)
    mem.save_json(a.out)
    if a.graphml:
        mem.save_graphml(a.graphml)
    if a.html:
        from .viz import export_html
        export_html(mem.to_networkx(), a.html)
        print(f"html: {a.html}")
    if a.store == "neo4j":
        store = get_store("neo4j")
        g = mem.to_networkx()
        for n, at in g.nodes(data=True):
            at = dict(at)
            store.upsert_node(n, at.pop("type"), at.pop("name"), **at)
        for u, v, at in g.edges(data=True):
            at = dict(at)
            store.upsert_edge(u, v, at.pop("rel"), **at)
        store.close()
        print("mirrored graph to Neo4j")
    s = graph_stats(mem.to_networkx())
    print(f"graph: {s['nodes']} nodes, {s['edges']} edges -> {a.out}")


def cmd_ask(a):
    from .query import graph_answer, vector_answer
    from .vector import Retriever
    g = _load(a.graph)
    if a.mode == "graph":
        ans = graph_answer(g, a.question)
        if not ans.items and not ans.counts and ans.parsed and not ans.parsed.constraints:
            print("(no entity linked; falling back to vector retrieval)")
            ans = vector_answer(Retriever.from_graph(g), a.question, a.k)
    else:
        ans = vector_answer(Retriever.from_graph(g, a.retriever), a.question, a.k)
    text = ans.text
    if a.llm or (a.llm is None and llm_available()):
        from .llm import compose_answer
        text = compose_answer(a.question, ans.context, ans.text)
        print("[answer composed by Claude from the retrieved context]")
    else:
        print("[offline template answer]")
    print(text)
    if a.verbose and ans.context:
        print("\n--- context ---\n" + ans.context)


def cmd_stats(a):
    print(json.dumps(graph_stats(_load(a.graph)), indent=2))


def cmd_eval(a):
    from .evaluate import format_report, load_questions, run_eval
    papers = load_papers(a.input)
    g = build_graph(papers, OfflineExtractor()).to_networkx()
    res = run_eval(g, load_questions(a.questions), tuple(a.k), a.retriever)
    print(format_report(res))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


def cmd_viz(a):
    from .viz import export_html
    print(export_html(_load(a.graph), a.out))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="neurograph", description=__doc__)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download abstracts from the arXiv API")
    f.add_argument("--query", action="append", help="repeatable; default: 4 BCI queries")
    f.add_argument("--max-results", type=int, default=25)
    f.add_argument("--out", default="data/arxiv.jsonl")
    f.add_argument("--delay", type=float, default=3.0, help="seconds between requests")
    f.set_defaults(fn=cmd_fetch)

    b = sub.add_parser("build", help="extract entities and build the knowledge graph")
    b.add_argument("--input", default=DEFAULT_CORPUS)
    b.add_argument("--out", default=DEFAULT_GRAPH)
    b.add_argument("--graphml")
    b.add_argument("--html", help="also write an interactive HTML view")
    b.add_argument("--extractor", choices=["auto", "offline", "llm"], default="auto")
    b.add_argument("--store", choices=["networkx", "neo4j"], default="networkx")
    b.set_defaults(fn=cmd_build)

    q = sub.add_parser("ask", help="answer a question")
    q.add_argument("question")
    q.add_argument("--mode", choices=["graph", "vector"], default="graph")
    q.add_argument("--graph", default=DEFAULT_GRAPH)
    q.add_argument("--k", type=int, default=3)
    q.add_argument("--retriever", choices=["bm25", "tfidf"], default="bm25")
    q.add_argument("--llm", action=argparse.BooleanOptionalAction, default=None,
                   help="force/disable Claude for the final answer (default: on if ANTHROPIC_API_KEY set)")
    q.add_argument("-v", "--verbose", action="store_true")
    q.set_defaults(fn=cmd_ask)

    s = sub.add_parser("stats", help="graph statistics")
    s.add_argument("--graph", default=DEFAULT_GRAPH)
    s.set_defaults(fn=cmd_stats)

    e = sub.add_parser("eval", help="vector vs graph on the multi-hop question set (offline)")
    e.add_argument("--input", default=DEFAULT_CORPUS)
    e.add_argument("--questions", default="data/eval_questions.json")
    e.add_argument("--k", type=int, nargs="+", default=[3, 5])
    e.add_argument("--retriever", choices=["bm25", "tfidf"], default="bm25")
    e.add_argument("--out", help="write full JSON results")
    e.set_defaults(fn=cmd_eval)

    v = sub.add_parser("viz", help="export interactive HTML")
    v.add_argument("--graph", default=DEFAULT_GRAPH)
    v.add_argument("--out", default="out/graph.html")
    v.set_defaults(fn=cmd_viz)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
