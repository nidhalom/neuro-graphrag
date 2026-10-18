"""Optional LLM answer composition (Claude via the official SDK). Offline template answers are the default."""
from __future__ import annotations

from .extract_llm import MODEL

SYSTEM = (
    "You answer research questions about EEG / brain-computer-interface literature. Use ONLY the provided "
    "context (facts extracted from paper abstracts). Cite paper ids in brackets. If the context does not "
    "contain the answer, say so."
)


def compose_answer(question: str, context: str, hint: str = "", client=None, model: str = MODEL) -> str:
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    user = f"Context:\n{context or '(empty)'}\n\nPrecomputed result: {hint or '(none)'}\n\nQuestion: {question}"
    resp = client.messages.create(model=model, max_tokens=16000, system=SYSTEM,
                                  messages=[{"role": "user", "content": user}])
    if getattr(resp, "stop_reason", None) == "refusal":
        return "(model refused to answer)"
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
