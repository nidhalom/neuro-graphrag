"""Offline evaluation: vector (BM25 top-k) vs GraphRAG on multi-hop questions with known answers."""
from __future__ import annotations

import json
from pathlib import Path

from .query import Answer, graph_answer, vector_answer
from .vector import Retriever


def _norm(x) -> str:
    return str(x).strip().lower()


def score(answer: Answer, qa: dict) -> dict:
    if qa["answer_type"] == "counts":
        gold = {f"{_norm(k)}={v}" for k, v in qa["answer"].items()}
        pred = {f"{_norm(k)}={v}" for k, v in (answer.counts or {}).items()}
    else:
        gold = {_norm(a) for a in qa["answer"]}
        pred = {_norm(a) for a in answer.items}
    tp = len(gold & pred)
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(gold) if gold else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "exact": pred == gold}


def run_eval(g, questions: list[dict], k_values=(3, 5), retriever_kind: str = "bm25") -> dict:
    retr = Retriever.from_graph(g, retriever_kind)
    systems = {"graph": lambda q: graph_answer(g, q)}
    for k in k_values:
        systems[f"vector(k={k})"] = (lambda q, k=k: vector_answer(retr, q, k))
    rows = []
    for qa in questions:
        row = {"id": qa["id"], "question": qa["question"], "hops": qa["hops"], "results": {}}
        for name, fn in systems.items():
            ans = fn(qa["question"])
            row["results"][name] = {**score(ans, qa), "items": ans.items if qa["answer_type"] != "counts"
                                    else ans.counts}
        rows.append(row)
    summary = {}
    for name in systems:
        rs = [r["results"][name] for r in rows]
        summary[name] = {"mean_f1": sum(r["f1"] for r in rs) / len(rs),
                         "exact_match": sum(r["exact"] for r in rs), "n": len(rs)}
    return {"summary": summary, "rows": rows, "retriever": retriever_kind}


def format_report(result: dict) -> str:
    names = list(result["summary"])
    lines = ["| id | hops | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for r in result["rows"]:
        lines.append(f"| {r['id']} | {r['hops']} | " +
                     " | ".join(f"{r['results'][n]['f1']:.2f}" for n in names) + " |")
    lines.append("| **mean F1** | | " + " | ".join(f"**{result['summary'][n]['mean_f1']:.2f}**" for n in names) + " |")
    lines.append("| **exact** | | " + " | ".join(
        f"**{result['summary'][n]['exact_match']}/{result['summary'][n]['n']}**" for n in names) + " |")
    return "\n".join(lines)


def load_questions(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
