"""Offline extractor: curated dictionary + regex for channel counts, sampling rates and metrics."""
from __future__ import annotations

import re

from . import schema as S
from .ontology import family_of, find_mentions
from .schema import MethodMention, MetricValue, Paper, PaperExtraction

_NUM = r"(\d+(?:\.\d+)?)"
_FRAC = r"(0?\.\d+)"

# (metric name, unit, compiled patterns). Each pattern has exactly one numeric group.
_METRIC_PATTERNS: list[tuple[str, str | None, list[re.Pattern]]] = [
    ("accuracy", "%", [
        re.compile(rf"accuracy(?:\s+(?:of|was|is|reached|reaching|=)|:)?\s*{_NUM}\s*%", re.I),
        re.compile(rf"{_NUM}\s*%\s+(?:mean\s+|average\s+|classification\s+)?accuracy", re.I),
    ]),
    ("kappa", None, [re.compile(rf"kappa(?:\s+(?:of|was|is|=)|:)?\s*{_FRAC}", re.I)]),
    ("auc", None, [re.compile(rf"\bAUC(?:\s+(?:of|was|is|=)|:)?\s*{_FRAC}")]),
    ("f1", None, [re.compile(rf"\bF1(?:-score)?(?:\s+(?:of|was|is|=)|:)?\s*{_FRAC}", re.I)]),
    ("itr", "bits/min", [re.compile(rf"{_NUM}\s*bits\s*/\s*min", re.I)]),
]

_CHANNELS = [
    re.compile(r"(\d{1,3})[- ](?:EEG\s+)?channels?\b", re.I),
    re.compile(r"(\d{1,3})\s+(?:EEG\s+)?electrodes\b", re.I),
    re.compile(r"\b(?:channels?|electrodes)\s*[:=]\s*(\d{1,3})\b", re.I),
]
_SAMPLING = [
    re.compile(rf"sampl\w*\s+(?:rate\s+|frequency\s+)?(?:of\s+|at\s+|=\s*|:\s*)?{_NUM}\s*Hz", re.I),
    re.compile(rf"{_NUM}\s*Hz\s+sampl\w*", re.I),
]


def _first_int(patterns, text) -> int | None:
    for rx in patterns:
        m = rx.search(text)
        if m:
            return int(m.group(1))
    return None


def extract_text(text: str) -> PaperExtraction:
    mentions = find_mentions(text)
    methods, datasets, paradigms, seen = [], [], [], set()
    for m in mentions:
        key = (m.type, m.name)
        if key in seen:
            continue
        seen.add(key)
        if m.type == S.METHOD:
            methods.append(MethodMention(name=m.name, family=family_of(m.name)))
        elif m.type == S.DATASET:
            datasets.append(m.name)
        elif m.type == S.PARADIGM:
            paradigms.append(m.name)

    metrics = []
    for name, unit, pats in _METRIC_PATTERNS:
        vals = [float(g) for rx in pats for g in rx.findall(text)]
        if vals:
            metrics.append(MetricValue(name=name, value=max(vals), unit=unit))

    sr = None
    for rx in _SAMPLING:
        m = rx.search(text)
        if m:
            sr = float(m.group(1))
            break
    return PaperExtraction(methods=methods, datasets=datasets, paradigms=paradigms,
                           metrics=metrics, channels=_first_int(_CHANNELS, text), sampling_rate_hz=sr)


class OfflineExtractor:
    name = "offline"

    def extract(self, paper: Paper) -> PaperExtraction:
        return extract_text(f"{paper.title}. {paper.abstract}")
