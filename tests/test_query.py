from neurograph.query import graph_answer, parse_question, vector_answer
from neurograph.vector import Retriever


def test_parse_intents():
    assert parse_question("Count papers per method on motor imagery.").op == "count_by"
    p = parse_question("Which methods are used for both motor imagery and P300?")
    assert p.target == "Method" and p.both and len(p.constraints) == 2
    assert parse_question("How are fuzzy classifier and EEGNet connected?").op == "path"
    assert parse_question("Which paper reports the highest accuracy on BCI Competition IV 2a?").op == "argmax"


def test_family_and_paradigm_two_hops(graph):
    a = graph_answer(graph, "Which papers use a spatial filter method on motor imagery?")
    assert a.items == ["syn-01", "syn-03", "syn-12"]  # xDAWN (P300) excluded


def test_family_filter_on_methods(graph):
    assert graph_answer(graph, "Which CNN methods were evaluated on BCI Competition IV 2a?").items == ["EEGNet"]


def test_bridging_methods(graph):
    a = graph_answer(graph, "Which methods are used for both motor imagery and P300?")
    assert a.items == ["EEGNet", "LDA", "Riemannian tangent space", "SVM"]


def test_aggregation_and_argmax(graph):
    a = graph_answer(graph, "Count papers per method on motor imagery.")
    assert a.counts["CSP"] == 2 and a.counts["Transformer"] == 1
    assert graph_answer(graph, "Which paper reports the highest accuracy on BCI Competition IV 2a?").items == ["syn-11"]


def test_path_finding(graph):
    a = graph_answer(graph, "How are fuzzy classifier and EEGNet connected?")
    assert a.items == ["motor imagery"] and len(a.paths[0]) == 5


def test_vector_baseline_misses_set_logic(graph):
    r = Retriever.from_graph(graph)
    a = vector_answer(r, "Which papers use both EEGNet and SVM?", k=3)
    assert len(a.items) == 3  # top-k, no conjunction
    assert graph_answer(graph, "Which papers use both EEGNet and SVM?").items == ["syn-08"]


def test_retrievers_rank_relevant_first(graph):
    for kind in ("bm25", "tfidf"):
        hits = Retriever.from_graph(graph, kind).search("Transformer encoder attention", 1)
        assert hits[0][0] == "syn-11"
