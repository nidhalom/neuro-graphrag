"""Curated BCI/EEG dictionary used for offline extraction and for linking entities in questions.

Aliases are regex fragments; a literal space matches any run of whitespace or hyphens.
Aliases of 1-2 letters are matched case-sensitively, everything else case-insensitively.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from . import schema as S


@dataclass(frozen=True)
class Entry:
    type: str
    name: str
    aliases: tuple[str, ...]
    family: str | None = None  # for methods


@dataclass(frozen=True)
class Mention:
    type: str
    name: str
    start: int
    end: int


def _e(t, name, aliases, family=None):
    return Entry(t, name, tuple(aliases), family)


M, D, P, F, X = S.METHOD, S.DATASET, S.PARADIGM, S.FAMILY, S.METRIC

ENTRIES: tuple[Entry, ...] = (
    # ---- methods ----
    _e(M, "CSP", ["CSP", "common spatial patterns?"], "spatial filter"),
    _e(M, "FBCSP", ["FBCSP", "filter bank CSP", "filter bank common spatial patterns?"], "spatial filter"),
    _e(M, "xDAWN", ["xDAWN"], "spatial filter"),
    _e(M, "EEGNet", ["EEGNet"], "CNN"),
    _e(M, "ShallowConvNet", ["ShallowConvNet", "Shallow ConvNet"], "CNN"),
    _e(M, "DeepConvNet", ["DeepConvNet", "Deep ConvNet"], "CNN"),
    _e(M, "LDA", ["LDA", "linear discriminant analysis"], "classical ML"),
    _e(M, "SVM", ["SVMs?", "support vector machines?"], "classical ML"),
    _e(M, "Random Forest", ["random forests?"], "classical ML"),
    _e(M, "Fuzzy classifier", ["fuzzy classifiers?", "fuzzy inference system", "fuzzy rule-based classifier"],
       "fuzzy system"),
    _e(M, "Riemannian tangent space", ["riemannian tangent space", "tangent space", "riemannian classifier"],
       "Riemannian geometry"),
    _e(M, "CCA", ["CCA", "canonical correlation analysis"], "correlation-based"),
    _e(M, "TRCA", ["TRCA", "task-related component analysis"], "correlation-based"),
    _e(M, "Transformer", ["EEG Transformers?", "Transformers?"], "attention-based"),
    _e(M, "LSTM", ["LSTM", "long short-term memory"], "recurrent network"),
    # ---- method families (linked in questions; created in the graph via IS_A) ----
    _e(F, "spatial filter", ["spatial filters?", "spatial filtering"]),
    _e(F, "CNN", ["CNNs?", "convolutional neural networks?", "convolutional networks?"]),
    _e(F, "classical ML", ["classical machine learning", "classical ML", "traditional machine learning",
                           "classical classifiers?"]),
    _e(F, "fuzzy system", ["fuzzy systems?", "fuzzy logic"]),
    _e(F, "Riemannian geometry", ["riemannian geometry", "riemannian"]),
    _e(F, "correlation-based", ["correlation-based"]),
    _e(F, "attention-based", ["attention-based", "self-attention"]),
    _e(F, "recurrent network", ["recurrent neural networks?", "RNNs?"]),
    # ---- paradigms ----
    _e(P, "motor imagery", ["motor imagery", "imagined movements?", "MI"]),
    _e(P, "P300", ["P300", "P300 speller"]),
    _e(P, "SSVEP", ["SSVEPs?", "steady-state visual(?:ly)? evoked potentials?"]),
    _e(P, "ERP", ["ERPs?", "event-related potentials?"]),
    # ---- datasets ----
    _e(D, "BCI Competition IV 2a", [r"BCI (?:Competition )?IV (?:dataset )?(?:2a|IIa)", "BNCI 2014-001"]),
    _e(D, "BCI Competition IV 2b", [r"BCI (?:Competition )?IV (?:dataset )?(?:2b|IIb)", "BNCI 2014-004"]),
    _e(D, "BCI Competition III IVa", [r"BCI (?:Competition )?III (?:dataset )?IVa"]),
    _e(D, "PhysioNet EEGMMIDB", ["EEGMMIDB", "PhysioNet EEG Motor Movement/Imagery",
                                 "EEG Motor Movement/Imagery Dataset"]),
    _e(D, "OpenBMI", ["OpenBMI"]),
    _e(D, "BNCI Horizon 2020 P300", ["BNCI Horizon (?:2020 )?P300", "BNCI Horizon 2020"]),
    _e(D, "Benchmark SSVEP", ["Benchmark SSVEP", "Tsinghua benchmark", "Tsinghua SSVEP benchmark"]),
    # ---- metrics (question linking only; values are extracted by regex) ----
    _e(X, "accuracy", ["accuracy", "accuracies"]),
    _e(X, "kappa", ["kappa"]),
    _e(X, "auc", ["AUC"]),
    _e(X, "f1", ["F1"]),
    _e(X, "itr", ["ITR", "information transfer rate"]),
)

BY_KEY = {(e.type, e.name): e for e in ENTRIES}


def _compile(alias: str) -> re.Pattern:
    pat = alias.replace(" ", r"[\s\-]+")
    flags = 0 if len(re.sub(r"[^A-Za-z]", "", alias)) <= 2 else re.IGNORECASE
    return re.compile(rf"(?<![A-Za-z0-9]){pat}(?![A-Za-z0-9])", flags)


@lru_cache(maxsize=1)
def _compiled() -> tuple[tuple[Entry, re.Pattern], ...]:
    return tuple((e, _compile(a)) for e in ENTRIES for a in e.aliases)


def find_mentions(text: str, types: tuple[str, ...] | None = None) -> list[Mention]:
    """Longest-match-first, non-overlapping entity linking. Returned in text order."""
    cands = []
    for e, rx in _compiled():
        if types and e.type not in types:
            continue
        for m in rx.finditer(text):
            cands.append(Mention(e.type, e.name, m.start(), m.end()))
    cands.sort(key=lambda m: (-(m.end - m.start), m.start))
    taken: list[Mention] = []
    for c in cands:
        if all(c.end <= t.start or c.start >= t.end for t in taken):
            taken.append(c)
    return sorted(taken, key=lambda m: m.start)


def family_of(method_name: str) -> str | None:
    e = BY_KEY.get((S.METHOD, method_name))
    return e.family if e else None


def canonical(ntype: str, name: str) -> str:
    """Map a free-text name (e.g. from an LLM) to the canonical dictionary name when possible."""
    ms = [m for m in find_mentions(name, (ntype,))]
    if ms and ms[0].start == 0 and ms[0].end >= len(name.strip()):
        return ms[0].name
    return name.strip()
