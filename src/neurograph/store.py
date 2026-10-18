"""Graph stores behind one interface: networkx (default, JSON/GraphML) and optional Neo4j (Cypher MERGE)."""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

import networkx as nx

from . import schema as S


class GraphStore(ABC):
    """Minimal write interface; queries always run on the networkx view from `to_networkx()`."""

    @abstractmethod
    def upsert_node(self, nid: str, ntype: str, name: str, **attrs) -> None: ...

    @abstractmethod
    def upsert_edge(self, src: str, dst: str, rel: str, **attrs) -> None: ...

    @abstractmethod
    def to_networkx(self) -> nx.DiGraph: ...

    def close(self) -> None:  # pragma: no cover - trivial
        pass


class NetworkxStore(GraphStore):
    def __init__(self, graph: nx.DiGraph | None = None):
        self.g = graph if graph is not None else nx.DiGraph()

    def upsert_node(self, nid, ntype, name, **attrs):
        self.g.add_node(nid, type=ntype, name=name, **{k: v for k, v in attrs.items() if v is not None})

    def upsert_edge(self, src, dst, rel, **attrs):
        self.g.add_edge(src, dst, rel=rel, **{k: v for k, v in attrs.items() if v is not None})

    def to_networkx(self):
        return self.g

    # ---- persistence ----
    def save_json(self, path: str | Path) -> None:
        data = {
            "nodes": [{"id": n, **a} for n, a in self.g.nodes(data=True)],
            "edges": [{"source": u, "target": v, **a} for u, v, a in self.g.edges(data=True)],
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "NetworkxStore":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        g = nx.DiGraph()
        for n in data["nodes"]:
            n = dict(n)
            g.add_node(n.pop("id"), **n)
        for e in data["edges"]:
            e = dict(e)
            g.add_edge(e.pop("source"), e.pop("target"), **e)
        return cls(g)

    def save_graphml(self, path: str | Path) -> None:
        h = nx.DiGraph()
        for n, a in self.g.nodes(data=True):
            h.add_node(n, **{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in a.items()})
        for u, v, a in self.g.edges(data=True):
            h.add_edge(u, v, **{k: (json.dumps(x) if isinstance(x, (list, dict)) else x) for k, x in a.items()})
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(h, p)


class Neo4jStore(GraphStore):
    """Optional adapter. Needs the `neo4j` driver and NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD."""

    def __init__(self, uri=None, user=None, password=None, driver=None, database: str | None = None):
        if driver is None:
            try:
                from neo4j import GraphDatabase
            except ImportError as e:
                raise RuntimeError("neo4j driver not installed: pip install 'neuro-graphrag[neo4j]'") from e
            uri = uri or os.environ.get("NEO4J_URI")
            user = user or os.environ.get("NEO4J_USER")
            password = password or os.environ.get("NEO4J_PASSWORD")
            if not (uri and user and password):
                raise RuntimeError("set NEO4J_URI, NEO4J_USER and NEO4J_PASSWORD")
            driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver, self.database = driver, database

    @staticmethod
    def _props(attrs: dict) -> dict:
        out = {}
        for k, v in attrs.items():
            if v is None:
                continue
            out[k] = json.dumps(v) if isinstance(v, dict) else v
        return out

    def _run(self, cypher: str, **params):
        with self.driver.session(database=self.database) as s:
            return list(s.run(cypher, **params))

    def upsert_node(self, nid, ntype, name, **attrs):
        if ntype not in S.NODE_TYPES:  # labels cannot be parameterised, so whitelist
            raise ValueError(f"bad node type {ntype}")
        self._run(f"MERGE (n:{ntype} {{id: $id}}) SET n += $props",
                  id=nid, props={"name": name, "type": ntype, **self._props(attrs)})

    def upsert_edge(self, src, dst, rel, **attrs):
        if rel not in S.RELATIONS:
            raise ValueError(f"bad relation {rel}")
        self._run(f"MATCH (a {{id: $src}}), (b {{id: $dst}}) MERGE (a)-[r:{rel}]->(b) SET r += $props",
                  src=src, dst=dst, props=self._props(attrs))

    def to_networkx(self):
        g = nx.DiGraph()
        for rec in self._run("MATCH (n) RETURN properties(n) AS p"):
            p = dict(rec["p"])
            g.add_node(p.pop("id"), **p)
        for rec in self._run("MATCH (a)-[r]->(b) RETURN a.id AS s, b.id AS t, type(r) AS rel, properties(r) AS p"):
            g.add_edge(rec["s"], rec["t"], rel=rec["rel"], **dict(rec["p"]))
        return g

    def close(self):
        self.driver.close()


def neo4j_configured() -> bool:
    try:
        import neo4j  # noqa: F401
    except ImportError:
        return False
    return all(os.environ.get(k) for k in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"))


def get_store(kind: str = "networkx") -> GraphStore:
    if kind == "networkx":
        return NetworkxStore()
    if kind == "neo4j":
        return Neo4jStore()
    raise ValueError(f"unknown store {kind!r}")
