"""Fetch real abstracts from the public arXiv API into JSONL (polite rate limiting)."""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

API = "http://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}
DEFAULT_QUERIES = ["motor imagery EEG", "EEGNet", "P300 speller", "SSVEP"]
USER_AGENT = "neuro-graphrag/0.1 (research tool; https://github.com/nidhalom/neuro-graphrag)"


def build_url(query: str, start: int = 0, max_results: int = 25) -> str:
    # quoted phrase on all fields; arXiv asks for >= 3 s between calls
    params = {"search_query": f'all:"{query}"', "start": start, "max_results": max_results,
              "sortBy": "relevance", "sortOrder": "descending"}
    return f"{API}?{urllib.parse.urlencode(params)}"


def _clean(s: str | None) -> str:
    return " ".join((s or "").split())


def parse_feed(xml_text: str, query: str | None = None) -> list[dict]:
    root = ET.fromstring(xml_text)
    out = []
    for e in root.findall("a:entry", NS):
        url = _clean(e.findtext("a:id", namespaces=NS))
        out.append({
            "id": "arxiv:" + url.rsplit("/abs/", 1)[-1],
            "title": _clean(e.findtext("a:title", namespaces=NS)),
            "abstract": _clean(e.findtext("a:summary", namespaces=NS)),
            "authors": [_clean(a.findtext("a:name", namespaces=NS)) for a in e.findall("a:author", NS)],
            "published": _clean(e.findtext("a:published", namespaces=NS))[:10],
            "url": url,
            "source": "arxiv",
            "query": query,
        })
    return out


def http_get(url: str, timeout: float = 30.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8")


def fetch(queries=None, max_results: int = 25, out: str | Path = "data/arxiv.jsonl",
          delay: float = 3.0, getter=http_get, sleep=time.sleep) -> list[dict]:
    """Download abstracts for each query, de-duplicate by arXiv id, write JSONL."""
    queries = queries or DEFAULT_QUERIES
    seen, papers = set(), []
    for i, q in enumerate(queries):
        if i:
            sleep(delay)
        for p in parse_feed(getter(build_url(q, 0, max_results)), q):
            if p["id"] not in seen and p["abstract"]:
                seen.add(p["id"])
                papers.append(p)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for p in papers:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return papers


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
