import pytest

from conftest import FIXTURE
from neurograph import schema as S
from neurograph.build import build_graph, graph_stats, load_papers
from neurograph.store import Neo4jStore, NetworkxStore


def test_schema_respected(graph):
    for u, v, a in graph.edges(data=True):
        src, dst = S.RELATIONS[a["rel"]]
        assert graph.nodes[u]["type"] == src and graph.nodes[v]["type"] == dst


def test_counts_and_attributes(graph):
    s = graph_stats(graph)
    assert s["nodes_by_type"]["Paper"] == 12
    assert s["edges_by_relation"]["USES_METHOD"] == 20
    p = graph.nodes["Paper:syn-02"]
    assert p["channels"] == 22 and p["sampling_rate_hz"] == 250
    assert graph["Paper:syn-02"]["Metric:accuracy"]["value"] == 76.8
    assert graph.has_edge("Method:EEGNet", "MethodFamily:CNN")


def test_json_roundtrip_and_graphml(graph, tmp_path):
    st = NetworkxStore(graph)
    st.save_json(tmp_path / "g.json")
    g2 = NetworkxStore.load_json(tmp_path / "g.json").to_networkx()
    assert set(g2.nodes) == set(graph.nodes) and g2.number_of_edges() == graph.number_of_edges()
    st.save_graphml(tmp_path / "g.graphml")
    assert (tmp_path / "g.graphml").stat().st_size > 0


def test_idempotent_upsert():
    papers = load_papers(FIXTURE)
    st = build_graph(papers)
    n = st.to_networkx().number_of_edges()
    build_graph(papers, store=st)
    assert st.to_networkx().number_of_edges() == n


class _Sess:
    def __init__(self, runs):
        self.runs = runs

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def run(self, cypher, **params):
        self.runs.append((cypher, params))
        return []


class FakeDriver:
    def __init__(self):
        self.runs = []

    def session(self, database=None):
        return _Sess(self.runs)


def test_neo4j_adapter_emits_merge():
    d = FakeDriver()
    st = Neo4jStore(driver=d)
    st.upsert_node("Paper:a", "Paper", "a", title="t")
    st.upsert_node("Method:CSP", "Method", "CSP")
    st.upsert_edge("Paper:a", "Method:CSP", "USES_METHOD")
    assert len(d.runs) == 3 and all("MERGE" in c for c, _ in d.runs)
    with pytest.raises(ValueError):
        st.upsert_node("x", "Paper) DETACH DELETE (n", "x")
