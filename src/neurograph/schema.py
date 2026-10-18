"""Pydantic models shared by the offline and LLM extractors, plus graph schema constants."""
from __future__ import annotations

from pydantic import BaseModel, Field

# Node types
PAPER, METHOD, DATASET, PARADIGM, METRIC, FAMILY = (
    "Paper", "Method", "Dataset", "Paradigm", "Metric", "MethodFamily")
NODE_TYPES = (PAPER, METHOD, DATASET, PARADIGM, METRIC, FAMILY)

# Relation types: name -> (source type, target type)
USES_METHOD, EVALUATED_ON, STUDIES, REPORTS, IS_A = (
    "USES_METHOD", "EVALUATED_ON", "STUDIES", "REPORTS", "IS_A")
RELATIONS = {
    USES_METHOD: (PAPER, METHOD),
    EVALUATED_ON: (PAPER, DATASET),
    STUDIES: (PAPER, PARADIGM),
    REPORTS: (PAPER, METRIC),
    IS_A: (METHOD, FAMILY),
}


def node_id(ntype: str, name: str) -> str:
    return f"{ntype}:{name}"


class Paper(BaseModel):
    id: str
    title: str
    abstract: str
    authors: list[str] = Field(default_factory=list)
    published: str | None = None
    url: str | None = None
    source: str = "unknown"  # "arxiv" | "synthetic"
    query: str | None = None


class MethodMention(BaseModel):
    name: str
    family: str | None = None


class MetricValue(BaseModel):
    name: str  # accuracy | kappa | auc | f1 | itr
    value: float
    unit: str | None = None


class PaperExtraction(BaseModel):
    """What an extractor returns for one paper (this is also the JSON contract for the LLM)."""
    methods: list[MethodMention] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    paradigms: list[str] = Field(default_factory=list)
    metrics: list[MetricValue] = Field(default_factory=list)
    channels: int | None = None
    sampling_rate_hz: float | None = None
