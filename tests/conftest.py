from pathlib import Path

import pytest

from neurograph.build import build_graph, load_papers

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "fixture_abstracts.jsonl"
QUESTIONS = ROOT / "data" / "eval_questions.json"


@pytest.fixture(scope="session")
def graph():
    return build_graph(load_papers(FIXTURE)).to_networkx()
