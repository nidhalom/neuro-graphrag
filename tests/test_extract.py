import json

import pytest

from neurograph.extract_llm import LLMExtractionError, LLMExtractor, parse_json_object
from neurograph.extract_offline import extract_text
from neurograph.ontology import find_mentions
from neurograph.schema import Paper


def test_dictionary_and_regex():
    ex = extract_text("EEGNet and an SVM on BCI Competition IV 2a (22 EEG channels, sampled at 250 Hz) "
                      "for motor imagery give 76.8% accuracy and a kappa of 0.69.")
    assert {m.name for m in ex.methods} == {"EEGNet", "SVM"}
    assert {m.name: m.family for m in ex.methods}["EEGNet"] == "CNN"
    assert ex.datasets == ["BCI Competition IV 2a"]
    assert ex.paradigms == ["motor imagery"]
    assert ex.channels == 22 and ex.sampling_rate_hz == 250
    m = {x.name: x.value for x in ex.metrics}
    assert m == {"accuracy": 76.8, "kappa": 0.69}


def test_fbcsp_not_confused_with_csp_and_longest_match():
    names = [m.name for m in find_mentions("FBCSP then CSP; P300 speller; BNCI Horizon 2020 P300")]
    assert names == ["FBCSP", "CSP", "P300", "BNCI Horizon 2020 P300"]


def test_band_frequencies_are_not_sampling_rates():
    assert extract_text("Band-pass 8-30 Hz filtering.").sampling_rate_hz is None


class _Block:
    def __init__(self, text):
        self.type, self.text = "text", text


class _Resp:
    def __init__(self, text, stop="end_turn"):
        self.content, self.stop_reason = [_Block(text)], stop


class _Messages:
    def __init__(self, text, stop):
        self.text, self.stop, self.calls = text, stop, []

    def create(self, **kw):
        self.calls.append(kw)
        return _Resp(self.text, self.stop)


class _Client:
    def __init__(self, text, stop="end_turn"):
        self.messages = _Messages(text, stop)


PAPER = Paper(id="x", title="t", abstract="a")


def test_llm_extractor_validates_and_canonicalises():
    payload = {"methods": [{"name": "common spatial patterns", "family": None}],
               "datasets": ["BCI IV 2a"], "paradigms": ["Motor Imagery"],
               "metrics": [{"name": "accuracy", "value": 70.0, "unit": "%"}], "channels": 22}
    c = _Client("```json\n" + json.dumps(payload) + "\n```")
    ex = LLMExtractor(client=c).extract(PAPER)
    assert ex.methods[0].name == "CSP" and ex.methods[0].family == "spatial filter"
    assert ex.datasets == ["BCI Competition IV 2a"] and ex.paradigms == ["motor imagery"]
    assert c.messages.calls[0]["model"] == "claude-opus-5-5"


def test_llm_extractor_rejects_bad_json_and_refusal():
    with pytest.raises(LLMExtractionError):
        LLMExtractor(client=_Client('{"methods": "nope"}')).extract(PAPER)
    with pytest.raises(LLMExtractionError):
        LLMExtractor(client=_Client("", stop="refusal")).extract(PAPER)
    assert parse_json_object('noise {"a": 1} tail') == {"a": 1}


def test_mi_abbreviation_needs_the_long_form():
    from neurograph.ontology import find_mentions
    mutual = "Channels were ranked by mutual information (MI) before classification of P300 epochs."
    assert [m.name for m in find_mentions(mutual) if m.type == "Paradigm"] == ["P300"]
    motor = "We decode motor imagery (MI) from 22 channels; MI trials last 4 s."
    assert {m.name for m in find_mentions(motor) if m.type == "Paradigm"} == {"motor imagery"}


def test_metric_skips_the_baseline_value():
    from neurograph.extract_offline import extract_text
    ex = extract_text("Our model reaches an accuracy of 71.2% versus 84.5% accuracy for the baseline EEGNet.")
    assert [(m.name, m.value) for m in ex.metrics] == [("accuracy", 71.2)]
    ex = extract_text("The proposed method obtained 82.0% accuracy, compared with 75.0% accuracy for CSP+LDA.")
    assert [(m.name, m.value) for m in ex.metrics] == [("accuracy", 82.0)]
