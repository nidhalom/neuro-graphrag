import json

from conftest import QUESTIONS
from neurograph.evaluate import format_report, load_questions, run_eval, score
from neurograph.fetch import build_url, fetch, parse_feed
from neurograph.query import Answer
from neurograph.viz import export_html, graph_to_vis

ATOM = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><id>http://arxiv.org/abs/0000.00001v1</id><title>  Test
 title </title><summary> An abstract
 here. </summary><published>2020-01-02T00:00:00Z</published>
<author><name>Test Author</name></author></entry></feed>"""


def test_eval_runner_graph_beats_vector(graph):
    res = run_eval(graph, load_questions(QUESTIONS))
    s = res["summary"]
    assert s["graph"]["mean_f1"] > s["vector(k=3)"]["mean_f1"]
    assert s["graph"]["n"] == 10
    assert "mean F1" in format_report(res)


def test_score_counts_and_sets():
    a = Answer("x", "q", None, items=["A", "b"])
    assert score(a, {"answer_type": "set", "answer": ["a", "c"]})["f1"] == 0.5
    c = Answer("x", "q", None, counts={"CSP": 2})
    assert score(c, {"answer_type": "counts", "answer": {"csp": 2}})["exact"]


def test_fetch_parses_and_rate_limits(tmp_path):
    sleeps = []
    out = tmp_path / "a.jsonl"
    papers = fetch(["q1", "q2"], 5, out, delay=3.0, getter=lambda url: ATOM, sleep=sleeps.append)
    assert sleeps == [3.0]  # one polite pause between two queries
    assert len(papers) == 1  # de-duplicated across queries
    rec = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert rec["title"] == "Test title" and rec["id"] == "arxiv:0000.00001v1"
    assert "export.arxiv.org/api/query" in build_url("EEGNet")
    assert parse_feed(ATOM)[0]["authors"] == ["Test Author"]


def test_html_export(graph, tmp_path):
    p = export_html(graph, tmp_path / "g.html")
    html = p.read_text(encoding="utf-8")
    assert "vis-network" in html and "Paper:syn-01" in html
    assert len(graph_to_vis(graph)["nodes"]) == graph.number_of_nodes()
