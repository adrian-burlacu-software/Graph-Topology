"""Run the self-contained V682 ontological reasoning experiment."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from .ontology import Ontology, build_demo_ontology, facts_to_documents


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "output"


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _visualization(ontology: Ontology, path: Path) -> None:
    proofs = ontology.all_proofs()
    explicit = set(ontology.explicit_facts)
    node_kind = {
        **{node: "entity" for node in ontology.entities},
        **{node: "type" for node in ontology.types},
        **{node: "property" for node in ontology.properties},
    }
    graph = {
        "nodes": [{"id": node, "kind": node_kind.get(node, "unknown")} for node in sorted(node_kind)],
        "edges": [
            {
                **fact.as_dict(),
                "kind": "DIRECT" if fact in explicit else "INFERRED",
                "proof": proof.as_dict(),
            }
            for fact, proof in sorted(proofs.items())
        ],
    }
    payload = json.dumps(graph).replace("</", "<\\/")
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>V682 Canonical Ontology</title>
<style>
body{{font:14px system-ui,sans-serif;margin:0;display:grid;grid-template-columns:250px 1fr 340px;height:100vh;color:#172033}}
aside{{padding:16px;border-right:1px solid #d5dbe7;overflow:auto}} #detail{{border-left:1px solid #d5dbe7}}
h1{{font-size:18px;margin-top:0}} select{{width:100%;padding:6px}} #graph{{width:100%;height:100%;background:#fafcff}}
.node.entity{{fill:#b9e4ff}} .node.type{{fill:#d2f2c8}} .node.property{{fill:#ffe5aa}} .node.unknown{{fill:#e5e7eb}}
.edge.direct{{stroke:#2463b5;stroke-width:2}} .edge.inferred{{stroke:#b35d16;stroke-width:2;stroke-dasharray:7 4}}
.label{{font-size:12px;pointer-events:none}} .edge-label{{font-size:10px;fill:#475569;pointer-events:none}}
pre{{white-space:pre-wrap;word-break:break-word;background:#f4f7fb;padding:10px}} .hint{{color:#52627a}}
</style></head><body>
<aside><h1>V682 ontology</h1><p class="hint">Actual canonical facts and on-demand derivations.</p>
<label for="focus">Focused entity</label><select id="focus"><option value="">All ontology</option></select>
<p><span style="color:#2463b5">━</span> direct<br><span style="color:#b35d16">┄</span> inferred</p></aside>
<svg id="graph" viewBox="0 0 900 650" aria-label="Ontology graph"></svg>
<aside id="detail"><h1>Evidence</h1><p class="hint">Click an edge to inspect its proof.</p></aside>
<script>
const graph={payload}, svg=document.querySelector('#graph'), detail=document.querySelector('#detail'), focus=document.querySelector('#focus');
for(const n of graph.nodes) focus.add(new Option(n.id,n.id));
const esc=s=>String(s).replace(/[&<>"]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c]));
const proof=p=>p.kind==='DIRECT' ? `DIRECT\\n${{p.fact.subject}} --${{p.fact.relation}}--> ${{p.fact.object}}` :
  p.premises.map(proof).join('\\n+\\n')+`\\n+\\nrule: ${{p.rule}}\\ntherefore ${{p.fact.subject}} --${{p.fact.relation}}--> ${{p.fact.object}} [INFERRED]`;
function render(){{
  const chosen=focus.value, edges=chosen?graph.edges.filter(e=>e.subject===chosen||e.object===chosen):graph.edges;
  const ids=[...new Set(edges.flatMap(e=>[e.subject,e.object]))], nodes=graph.nodes.filter(n=>ids.includes(n.id));
  const positions=Object.fromEntries(nodes.map((n,i)=>[n.id,{{x:110+(i%4)*235,y:105+Math.floor(i/4)*145}}]));
  svg.replaceChildren(); const ns='http://www.w3.org/2000/svg';
  const add=(tag,attrs,text)=>{{const e=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));if(text)e.textContent=text;svg.append(e);return e;}};
  for(const edge of edges){{const a=positions[edge.subject],b=positions[edge.object];if(!a||!b)continue;
    const line=add('line',{{x1:a.x,y1:a.y,x2:b.x,y2:b.y,class:'edge '+edge.kind.toLowerCase()}});line.style.cursor='pointer';
    line.onclick=()=>detail.innerHTML='<h1>'+edge.kind+'</h1><pre>'+esc(proof(edge.proof))+'</pre>';
    add('text',{{x:(a.x+b.x)/2,y:(a.y+b.y)/2-5,class:'edge-label'}},edge.relation);
  }}
  for(const node of nodes){{const p=positions[node.id];add('circle',{{cx:p.x,cy:p.y,r:34,class:'node '+node.kind}});add('text',{{x:p.x,y:p.y+4,'text-anchor':'middle',class:'label'}},node.id);}}
  if(chosen) detail.innerHTML='<h1>Focused: '+esc(chosen)+'</h1><p>'+edges.length+' local relationship(s). Click an edge for evidence.</p>';
}}
focus.onchange=render;render();
</script></body></html>"""
    path.write_text(document, encoding="utf-8")


def _benchmarks(ontology: Ontology) -> list[dict]:
    cases = [
        ("dog type mammal", ontology.query("dog", "type", "mammal"), "DIRECT"),
        ("mammal is_a organism", ontology.query("mammal", "is_a", "organism"), "INFERRED"),
        ("dog type organism", ontology.query("dog", "type", "organism"), "INFERRED"),
        ("dog type mineral", ontology.query("dog", "type", "mineral"), "UNVERIFIED"),
        ("mammal type plant", ontology.query("mammal", "type", "plant"), "UNVERIFIED"),
        ("alias type_of dog mammal", ontology.query("dog", "type_of", "mammal"), "DIRECT"),
        ("property inheritance", ontology.query("dog", "has_property", "warm_blooded"), "INFERRED"),
        ("grounded question", ontology.query_natural_language("is dog an organism?"), "INFERRED"),
        ("grounded follow-up", ontology.query_natural_language("so is dog an organism?"), "INFERRED"),
    ]
    return [{"name": name, "expected": expected, "actual": result.status, "passed": result.status == expected,
             "result": result.as_dict()} for name, result, expected in cases]


def _proof_lines(proof: dict) -> list[str]:
    if proof["kind"] == "DIRECT":
        fact = proof["fact"]
        alias = proof.get("source_relation")
        source = f" (normalized from {alias})" if alias and alias != fact["relation"] else ""
        return [f"{fact['subject']} --{fact['relation']}--> {fact['object']} [DIRECT]{source}"]
    lines: list[str] = []
    for premise in proof["premises"]:
        lines.extend(_proof_lines(premise))
    fact = proof["fact"]
    lines.append(f"{proof['rule']}; therefore {fact['subject']} --{fact['relation']}--> {fact['object']} [INFERRED]")
    return lines


def run(output: Path = DEFAULT_OUTPUT) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    ontology = build_demo_ontology()
    inferred = ontology.derive()
    benchmarks = _benchmarks(ontology)
    evaluation = {
        "passed": sum(case["passed"] for case in benchmarks),
        "total": len(benchmarks),
        "cases": benchmarks,
        "stats": ontology.stats(),
    }
    _json(output / "ontology.json", ontology.ontology_document())
    _json(output / "inferred_facts.json", facts_to_documents(inferred.items()))
    _json(output / "proofs.json", facts_to_documents(ontology.all_proofs().items()))
    _json(output / "evaluation.json", evaluation)
    _visualization(ontology, output / "ontology.html")
    return evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.output)
    stats = result["stats"]
    print(f"entities: {stats['entities']}")
    print(f"canonical relations: {stats['canonical_relations']}")
    print(f"aliases: {stats['aliases']}")
    print(f"explicit facts: {stats['explicit_facts']}")
    print(f"inferred facts: {stats['inferred_facts']}")
    print(f"rules: {stats['rules']}")
    print(f"average proof depth: {stats['average_proof_depth']}")
    for case in result["cases"][:3]:
        print(f"{case['name']:<24} {'PASS' if case['passed'] else 'FAIL'}")
    dog = next(case for case in result["cases"] if case["name"] == "dog type organism")
    print("proof for dog type organism:")
    for line in _proof_lines(dog["result"]["proof"]):
        print(line)
    print(f"tests: {result['passed']}/{result['total']} passed")


if __name__ == "__main__":
    main()
