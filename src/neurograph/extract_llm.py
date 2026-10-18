"""LLM extractor: Claude via the official `anthropic` SDK, JSON validated with pydantic."""
from __future__ import annotations

import json
import os
import re

from pydantic import ValidationError

from . import ontology
from . import schema as S
from .schema import MethodMention, Paper, PaperExtraction

MODEL = "claude-opus-5-5"

SYSTEM = (
    "You extract structured facts from EEG / brain-computer-interface paper abstracts. "
    "Only report what the abstract explicitly states; never guess. Reply with ONE JSON object and nothing else."
)

PROMPT = """Extract from the abstract below. JSON schema:
{{
  "methods": [{{"name": str, "family": str|null}}],
  "datasets": [str],
  "paradigms": [str],
  "metrics": [{{"name": "accuracy|kappa|auc|f1|itr", "value": number, "unit": str|null}}],
  "channels": int|null,
  "sampling_rate_hz": number|null
}}
Notes: methods are algorithms/models used (e.g. CSP, EEGNet, SVM) with a family such as "spatial filter",
"CNN" or "classical ML"; datasets are named datasets (e.g. "BCI Competition IV 2a"); paradigms are e.g.
"motor imagery", "P300", "SSVEP", "ERP".

Title: {title}
Abstract: {abstract}"""


class LLMExtractionError(RuntimeError):
    pass


def parse_json_object(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise LLMExtractionError("no JSON object in model output")
    return json.loads(text[start:end + 1])


def _normalise(ex: PaperExtraction) -> PaperExtraction:
    """Map names to canonical dictionary names and fill in missing families."""
    methods, seen = [], set()
    for m in ex.methods:
        name = ontology.canonical(S.METHOD, m.name)
        if name in seen:
            continue
        seen.add(name)
        methods.append(MethodMention(name=name, family=m.family or ontology.family_of(name)))
    return ex.model_copy(update={
        "methods": methods,
        "datasets": list(dict.fromkeys(ontology.canonical(S.DATASET, d) for d in ex.datasets)),
        "paradigms": list(dict.fromkeys(ontology.canonical(S.PARADIGM, p) for p in ex.paradigms)),
    })


class LLMExtractor:
    name = "llm"

    def __init__(self, client=None, model: str = MODEL):
        if client is None:
            import anthropic  # optional dependency
            client = anthropic.Anthropic()
        self.client, self.model = client, model

    def extract(self, paper: Paper) -> PaperExtraction:
        resp = self.client.messages.create(
            model=self.model, max_tokens=16000, system=SYSTEM,
            messages=[{"role": "user", "content": PROMPT.format(title=paper.title, abstract=paper.abstract)}],
        )
        if getattr(resp, "stop_reason", None) == "refusal":
            raise LLMExtractionError(f"model refused to process {paper.id}")
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        try:
            return _normalise(PaperExtraction.model_validate(parse_json_object(text)))
        except (ValidationError, json.JSONDecodeError) as e:
            raise LLMExtractionError(f"invalid extraction for {paper.id}: {e}") from e


def llm_available() -> bool:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True
