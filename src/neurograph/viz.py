"""Export the graph as a self-contained interactive HTML page (vis-network loaded from a CDN)."""
from __future__ import annotations

import json
from html import escape
from pathlib import Path

import networkx as nx

from . import schema as S

COLORS = {S.PAPER: "#4c78a8", S.METHOD: "#f58518", S.DATASET: "#54a24b", S.PARADIGM: "#b279a2",
          S.METRIC: "#9d9da1", S.FAMILY: "#e45756"}

TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>neuro-graphrag knowledge graph</title>
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
body{margin:0;font:14px system-ui,sans-serif;display:flex;flex-direction:column;height:100vh}
header{padding:8px 12px;border-bottom:1px solid #ccc;display:flex;gap:14px;flex-wrap:wrap;align-items:center}
label{cursor:pointer}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px}
#net{flex:1}input{padding:3px 6px}
</style></head><body>
<header><strong>neuro-graphrag</strong><span id="legend"></span>
<input id="q" placeholder="search node..."><span id="info"></span></header>
<div id="net"></div>
<script>
const DATA = __DATA__;
const COLORS = __COLORS__;
const nodes = new vis.DataSet(DATA.nodes), edges = new vis.DataSet(DATA.edges);
const net = new vis.Network(document.getElementById('net'), {nodes, edges}, {
  nodes:{shape:'dot',font:{size:12}}, edges:{arrows:'to',font:{size:9,align:'middle'},smooth:false,color:{opacity:.6}},
  physics:{solver:'forceAtlas2Based',stabilization:{iterations:150}}, interaction:{hover:true}});
const legend = document.getElementById('legend');
Object.entries(COLORS).forEach(([t,c])=>{
  const l=document.createElement('label');
  l.innerHTML='<input type="checkbox" checked data-t="'+t+'"><span class="dot" style="background:'+c+'"></span>'+t+' ';
  legend.appendChild(l);});
legend.addEventListener('change',()=>{
  const off=new Set([...legend.querySelectorAll('input')].filter(i=>!i.checked).map(i=>i.dataset.t));
  nodes.update(DATA.nodes.map(n=>({id:n.id,hidden:off.has(n.group)})));});
document.getElementById('q').addEventListener('input',e=>{
  const s=e.target.value.toLowerCase(); if(!s)return;
  const hit=DATA.nodes.find(n=>n.label.toLowerCase().includes(s));
  if(hit){net.selectNodes([hit.id]);net.focus(hit.id,{scale:1.2,animation:true});}});
document.getElementById('info').textContent = DATA.nodes.length+' nodes, '+DATA.edges.length+' edges';
</script></body></html>
"""


def graph_to_vis(g: nx.DiGraph) -> dict:
    nodes = []
    for nid, a in g.nodes(data=True):
        t = a["type"]
        label = a["name"]
        title = f"<b>{escape(t)}</b>: {escape(a.get('title') or a['name'])}"
        deg = g.degree(nid)
        nodes.append({"id": nid, "label": label, "group": t, "title": title,
                      "color": COLORS.get(t, "#999"), "value": 2 + deg})
    edges = []
    for u, v, a in g.edges(data=True):
        lab = a["rel"] if a.get("value") is None else f"{a['rel']} {a['value']:g}"
        edges.append({"from": u, "to": v, "label": lab})
    return {"nodes": nodes, "edges": edges}


def export_html(g: nx.DiGraph, path: str | Path) -> Path:
    data = json.dumps(graph_to_vis(g), ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.replace("__DATA__", data).replace("__COLORS__", json.dumps(COLORS))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")
    return p
