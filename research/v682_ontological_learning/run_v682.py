"""Run V682 against the real focused semantic SQLite graph."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .ontology import Fact, Proof, SemanticGraph, facts_document


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
DEFAULT_DATABASE = REPOSITORY_ROOT / "data" / "v673_focused_semantic.sqlite"
DEFAULT_OUTPUT = HERE / "output"


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _proof_text(proof: dict[str, Any]) -> str:
    if proof["kind"] == "DIRECT":
        fact = proof["fact"]
        return f"DIRECT ({proof['source']}): {fact['subject']} --{fact['relation']}--> {fact['object']}"
    premises = "\n".join(_proof_text(item) for item in proof["premises"])
    rule = proof["rule"]
    fact = proof["fact"]
    return (
        f"{premises}\nrule: {rule['left_relation']} + {rule['right_relation']} -> "
        f"{rule['result_relation']} (confidence {rule['confidence']})\n"
        f"therefore {fact['subject']} --{fact['relation']}--> {fact['object']} [INFERRED]"
    )


def _real_data_evaluation(graph: SemanticGraph, rules: list[dict[str, Any]],
                          inferred: dict[Fact, Proof]) -> dict[str, Any]:
    dog = graph.resolve_node("dog")
    mammal = graph.resolve_node("mammal")
    organism = graph.resolve_node("organism")
    original = []
    if dog and mammal and organism:
        for label, fact in (
            ("dog -> mammal", Fact(dog, "is_a", mammal)),
            ("mammal -> organism", Fact(mammal, "is_a", organism)),
            ("dog -> organism", Fact(dog, "is_a", organism)),
        ):
            proof = graph.prove(fact, rules)
            original.append({
                "query": label, "fact": fact.as_dict(), "status": "VERIFIED" if proof else "UNVERIFIED",
                "evidence_kind": proof.kind if proof else None, "proof": proof.as_dict() if proof else None,
            })
    selected = None
    for fact, proof in sorted(inferred.items()):
        if proof.depth == 1:
            selected = {
                "kind": "automatically_selected_real_two_hop_chain",
                "result": fact.as_dict(),
                "proof": proof.as_dict(),
            }
            break
    if selected is None:
        for rule in rules:
            sample = rule["samples"][0] if rule["samples"] else None
            if sample:
                selected = {
                    "kind": "observed_real_two_hop_chain_no_new_derivation_available",
                    "rule": {key: rule[key] for key in ("left_relation", "right_relation", "result_relation")},
                    "sample": sample,
                }
                break
    dog_queries = [
        graph.query_natural_language("is dog a mammal?", rules),
        graph.query_natural_language("is dog an organism?", rules),
        graph.query_natural_language("so is dog an organism?", rules),
    ]
    return {
        "original_dog_chain": {
            "available": bool(dog and mammal and organism),
            "note": (
                "Exact links are reported from the real graph; absent links are not fabricated."
                if dog and mammal and organism else
                "At least one of dog, mammal, organism is absent from the focused graph."
            ),
            "results": original,
        },
        "automatic_equivalent_chain": selected,
        "grounded_queries": dog_queries,
    }


def _visualization(graph: SemanticGraph, inferred: dict[Fact, Proof], discovery: dict[str, Any],
                   relations: list[dict[str, Any]], evaluation: dict[str, Any], path: Path) -> None:
    degree = Counter()
    for edge in graph.edges:
        degree.update((edge.fact.subject, edge.fact.object))
    graph_edges = [
        {**edge.fact.as_dict(), "kind": "DIRECT", "source": edge.source,
         "importance": degree[edge.fact.subject] + degree[edge.fact.object]}
        for edge in graph.edges
    ] + [
        {**fact.as_dict(), "kind": "INFERRED", "proof": proof.as_dict(),
         "importance": degree[fact.subject] + degree[fact.object]}
        for fact, proof in inferred.items()
    ]
    payload = {
        "stats": graph.graph_stats(),
        "queryRelations": sorted(
            relation for relation, phrases in graph.relation_phrases.items()
            if "is a" in phrases.lower()
        ),
        "nodes": [{**node, "degree": degree[node_id]} for node_id, node in graph.nodes.items()],
        "edges": graph_edges,
        "rules": [{key: item[key] for key in (
            "left_relation", "right_relation", "result_relation", "status", "support",
            "precision", "confidence", "contradictions",
        )} for item in discovery["rules"]],
        "relationModels": relations,
        "evaluation": evaluation,
    }
    encoded = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>V682 Real Semantic Knowledge Globe</title>
<style>
:root{{color-scheme:dark;font:13px system-ui,sans-serif;background:#07111f;color:#dce9ff}}*{{box-sizing:border-box}}
body{{margin:0;display:grid;grid-template-columns:310px minmax(500px,1fr) 350px;height:100vh;overflow:hidden}}
aside{{padding:14px;overflow:auto;background:#0b1728;border-color:#233952;border-style:solid}}#controls{{border-width:0 1px 0 0}}#detail{{border-width:0 0 0 1px}}
h1{{margin:0 0 10px;font-size:18px}}h2{{font-size:14px;margin:17px 0 6px}}p{{line-height:1.4}}label{{display:block;margin-top:8px}}input,button,select{{background:#10243b;color:#e8f3ff;border:1px solid #3a5c7d;border-radius:4px;padding:6px}}input,select{{width:100%}}button{{cursor:pointer;margin-top:5px}}canvas{{width:100%;height:100%;background:radial-gradient(circle at 50% 44%,#183653,#081321 68%);touch-action:none}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:6px}}.metric{{padding:7px;background:#10243b;border-radius:4px}}.muted{{color:#98aec9}}.legend span{{padding:2px 5px;margin:2px;display:inline-block;border-radius:3px}}#relations{{max-height:200px;overflow:auto;background:#091727;padding:4px}}.rel{{display:flex;gap:6px;align-items:center}}.rel input{{width:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#07101c;padding:10px;border-radius:4px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:4px;text-align:left;border-bottom:1px solid #253b54;font-size:11px}}.accepted{{color:#8ee89b}}.rejected{{color:#ff9f9f}}
</style></head><body>
<aside id="controls"><h1>V682 real knowledge globe</h1><p class="muted">Full focused SQLite graph embedded. Initial view uses level-of-detail; focus/search reveals actual neighborhoods.</p>
<div class="grid" id="stats"></div><h2>Graph query</h2><input id="query" value="is dog an organism?" aria-label="Graph query"><button id="ask">Query graph evidence</button><div id="answer"></div>
<h2>Focus and filtering</h2><input id="search" placeholder="Search node or relation" aria-label="Search"><button id="focus">Focus result</button><button id="clear">Clear focus</button>
<label><input id="direct" type="checkbox" checked> Direct edges</label><label><input id="inferred" type="checkbox" checked> Inferred edges</label>
<div id="relations"></div><p class="muted">Drag empty space: rotate. Shift-drag: pan. Drag a node: reposition. Wheel: zoom. Click node/edge: inspect.</p></aside>
<canvas id="globe" aria-label="Interactive 3D semantic graph"></canvas>
<aside id="detail"><h1>Inspector</h1><p class="muted">Select a real graph node or edge.</p><h2>Strongest discovered rules</h2><table id="rules"><thead><tr><th>rule</th><th>support</th><th>precision</th></tr></thead><tbody></tbody></table></aside>
<script>
const data={encoded}, canvas=document.querySelector('#globe'), ctx=canvas.getContext('2d'), detail=document.querySelector('#detail');
const state={{yaw:.45,pitch:-.3,zoom:1,panX:0,panY:0,focus:'',drag:null,last:null,showDirect:true,showInferred:true,relations:new Set(data.edges.map(e=>e.relation))}};
const degrees=Object.fromEntries(data.nodes.map(n=>[n.id,n.degree])); const ids=data.nodes.map(n=>n.id).sort();
const pos={{}}; ids.forEach((id,i)=>{{const y=1-(i/(Math.max(ids.length-1,1)))*2,r=Math.sqrt(Math.max(0,1-y*y)),a=i*2.399963229728653;pos[id]={{x:Math.cos(a)*r,y,z:Math.sin(a)*r}};}});
const resize=()=>{{canvas.width=canvas.clientWidth*devicePixelRatio;canvas.height=canvas.clientHeight*devicePixelRatio;ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);render();}};addEventListener('resize',resize);resize();
const esc=s=>String(s).replace(/[&<>"]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c]));
function proof(p){{if(p.kind==='DIRECT')return `DIRECT (${{p.source}})\\n${{p.fact.subject}} --${{p.fact.relation}}--> ${{p.fact.object}}`;let r=p.rule;return p.premises.map(proof).join('\\n+\\n')+`\\n+\\nrule: ${{r.left_relation}} + ${{r.right_relation}} -> ${{r.result_relation}} (confidence ${{r.confidence}})\\ntherefore ${{p.fact.subject}} --${{p.fact.relation}}--> ${{p.fact.object}} [INFERRED]`;}}
function project(v){{let cy=Math.cos(state.yaw),sy=Math.sin(state.yaw),cp=Math.cos(state.pitch),sp=Math.sin(state.pitch);let x=v.x*cy-v.z*sy,z=v.x*sy+v.z*cy,y=v.y*cp-z*sp;return {{x:canvas.clientWidth/2+state.panX+x*250*state.zoom,y:canvas.clientHeight/2+state.panY-y*250*state.zoom,z}};}}
function visible(e){{return state.relations.has(e.relation)&&(e.kind==='DIRECT'?state.showDirect:state.showInferred)&&(!state.focus||e.subject===state.focus||e.object===state.focus);}}
function color(s){{let n=0;for(const c of s)n=(n*31+c.charCodeAt(0))>>>0;return `hsl(${{n%360}} 68% 65%)`;}}
function render(){{ctx.clearRect(0,0,canvas.clientWidth,canvas.clientHeight);let edges=data.edges.filter(visible);if(!state.focus)edges=edges.sort((a,b)=>b.importance-a.importance).slice(0,750);let shown=new Set(edges.flatMap(e=>[e.subject,e.object]));let ps=Object.fromEntries([...shown].map(id=>[id,project(pos[id])]));state.drawn={{edges,ps}};
 for(const e of edges.sort((a,b)=>((ps[a.subject]?.z||0)+(ps[a.object]?.z||0))-((ps[b.subject]?.z||0)+(ps[b.object]?.z||0)))){{let a=ps[e.subject],b=ps[e.object];if(!a||!b)continue;ctx.beginPath();ctx.setLineDash(e.kind==='INFERRED'?[5,4]:[]);ctx.strokeStyle=e.kind==='INFERRED'?'#ffac52':color(e.relation);ctx.globalAlpha=.22;ctx.lineWidth=e.kind==='INFERRED'?1.5:1;ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();}}ctx.setLineDash([]);ctx.globalAlpha=1;
 for(const id of [...shown].sort((a,b)=>(ps[a].z-ps[b].z))){{let p=ps[id],n=data.nodes.find(x=>x.id===id),rad=Math.max(3,Math.min(13,3+Math.sqrt(degrees[id]||0)));ctx.beginPath();ctx.fillStyle=n.node_type==='synset'?'#a6f5c5':'#87c9ff';ctx.arc(p.x,p.y,rad,0,Math.PI*2);ctx.fill();if(state.focus===id||rad>8){{ctx.fillStyle='#e8f3ff';ctx.font='11px system-ui';ctx.fillText(n.label,p.x+rad+3,p.y+3);}}}}}}
function nearest(x,y){{let best=null,d=13;for(const [id,p] of Object.entries(state.drawn.ps||{{}})){{let q=Math.hypot(p.x-x,p.y-y);if(q<d){{best={{kind:'node',id}};d=q;}}}}if(best)return best;for(const e of state.drawn.edges||[]){{let a=state.drawn.ps[e.subject],b=state.drawn.ps[e.object],dx=b.x-a.x,dy=b.y-a.y,t=Math.max(0,Math.min(1,((x-a.x)*dx+(y-a.y)*dy)/(dx*dx+dy*dy||1))),q=Math.hypot(x-(a.x+t*dx),y-(a.y+t*dy));if(q<5){{best={{kind:'edge',edge:e}};break;}}}}return best;}}
canvas.onpointerdown=e=>{{let p=nearest(e.offsetX,e.offsetY);state.drag=p?.kind==='node'?p:null;state.last={{x:e.offsetX,y:e.offsetY,shift:e.shiftKey}};canvas.setPointerCapture(e.pointerId);}};
canvas.onpointermove=e=>{{if(!state.last)return;let dx=e.offsetX-state.last.x,dy=e.offsetY-state.last.y;if(state.drag){{let p=pos[state.drag.id];p.x+=dx/(250*state.zoom);p.y-=dy/(250*state.zoom);}}else if(state.last.shift){{state.panX+=dx;state.panY+=dy;}}else{{state.yaw+=dx*.008;state.pitch=Math.max(-1.5,Math.min(1.5,state.pitch+dy*.008));}}state.last={{x:e.offsetX,y:e.offsetY,shift:e.shiftKey}};render();}};
canvas.onpointerup=e=>{{let picked=nearest(e.offsetX,e.offsetY);if(picked&&Math.abs(e.offsetX-state.last.x)<2&&Math.abs(e.offsetY-state.last.y)<2)inspect(picked);state.last=null;state.drag=null;}};
canvas.onwheel=e=>{{e.preventDefault();state.zoom=Math.max(.2,Math.min(4,state.zoom*(e.deltaY<0?1.12:.89)));render();}};
function inspect(x){{if(x.kind==='edge'){{let e=x.edge;detail.innerHTML='<h1>'+e.kind+' edge</h1><pre>'+esc(e.subject+' --'+e.relation+'--> '+e.object+(e.kind==='DIRECT'?'\\nsource: '+e.source:'\\n'+proof(e.proof)))+'</pre>';return;}}state.focus=x.id;let n=data.nodes.find(n=>n.id===x.id),edges=data.edges.filter(e=>e.subject===x.id||e.object===x.id),direct=edges.filter(e=>e.kind==='DIRECT').length,inferred=edges.length-direct;detail.innerHTML='<h1>'+esc(n.label)+'</h1><p>'+esc(x.id)+' · '+esc(n.node_type)+'</p><div class=\"grid\"><div class=\"metric\">'+direct+' direct</div><div class=\"metric\">'+inferred+' inferred</div></div><h2>Neighborhood</h2><pre>'+esc(edges.slice(0,120).map(e=>(e.subject===x.id?'→ ':'← ')+e.relation+' '+(e.subject===x.id?e.object:e.subject)+' ['+e.kind+']').join('\\n'))+'</pre>';render();}}
const stats=document.querySelector('#stats');for(const [k,v] of Object.entries({{entities:data.stats.entities,edges:data.stats.edges,relations:data.stats.unique_relations,direct:data.edges.filter(e=>e.kind==='DIRECT').length,inferred:data.edges.filter(e=>e.kind==='INFERRED').length,candidates:data.rules.length,accepted:data.rules.filter(r=>r.status==='ACCEPTED').length,rejected:data.rules.filter(r=>r.status==='REJECTED').length}}))stats.innerHTML+='<div class=\"metric\"><b>'+v+'</b><br><span class=\"muted\">'+k+'</span></div>';
const rel=document.querySelector('#relations'), relCounts={{}};data.edges.forEach(e=>relCounts[e.relation]=(relCounts[e.relation]||0)+1);Object.entries(relCounts).sort((a,b)=>b[1]-a[1]).forEach(([r,n])=>{{let l=document.createElement('label');l.className='rel';l.innerHTML='<input type=\"checkbox\" checked> '+esc(r)+' ('+n+')';l.firstChild.onchange=e=>{{e.target.checked?state.relations.add(r):state.relations.delete(r);render();}};rel.append(l);}});
document.querySelector('#direct').onchange=e=>{{state.showDirect=e.target.checked;render();}};document.querySelector('#inferred').onchange=e=>{{state.showInferred=e.target.checked;render();}};
document.querySelector('#focus').onclick=()=>{{let q=document.querySelector('#search').value.toLowerCase(), n=data.nodes.find(x=>x.id.toLowerCase()===q||x.label.toLowerCase()===q)||data.nodes.find(x=>x.id.toLowerCase().includes(q)||x.label.toLowerCase().includes(q));if(n)inspect({{kind:'node',id:n.id}});else{{let r=Object.keys(relCounts).find(x=>x.includes(q));if(r){{state.relations=new Set([r]);render();}}}}}};document.querySelector('#clear').onclick=()=>{{state.focus='';render();}};
const rules=document.querySelector('#rules tbody');data.rules.filter(r=>r.status==='ACCEPTED').slice(0,10).forEach(r=>rules.innerHTML+='<tr><td class=\"accepted\">'+esc(r.left_relation+' + '+r.right_relation+' → '+r.result_relation)+'</td><td>'+r.support+'</td><td>'+r.precision+'</td></tr>');
document.querySelector('#ask').onclick=()=>{{let q=document.querySelector('#query').value.toLowerCase().replace(/^(so|then|and)\\s*,?\\s*/,'').replace(/[?.! ]+$/,''),m=q.match(/^is\\s+(.+?)\\s+(?:a|an)\\s+(.+)$/);if(!m){{document.querySelector('#answer').textContent='UNVERIFIED: supported form is “is A a B?”';return;}}let node=t=>data.nodes.find(n=>n.label.toLowerCase()===t||n.normalized.toLowerCase()===t||n.id.toLowerCase()==='en:'+t);let a=node(m[1]),b=node(m[2]),edge=a&&b&&data.edges.find(e=>e.subject===a.id&&e.object===b.id&&data.queryRelations.includes(e.relation));document.querySelector('#answer').innerHTML=edge?'<pre>VERIFIED ['+edge.kind+']\\n'+esc(edge.proof?proof(edge.proof):(edge.subject+' --'+edge.relation+'--> '+edge.object+'\\nsource: '+edge.source))+'</pre>':'<pre>UNVERIFIED\\nNo direct fact or accepted-rule derivation is embedded for this query.</pre>';}};render();
</script></body></html>"""
    path.write_text(document, encoding="utf-8")


def run(database: Path = DEFAULT_DATABASE, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    graph = SemanticGraph(database)
    output.mkdir(parents=True, exist_ok=True)
    discovery = graph.discover_rules()
    accepted = [rule for rule in discovery["rules"] if rule["status"] == "ACCEPTED"]
    inferred = graph.infer(accepted)
    relations = graph.relation_models(discovery)
    evaluation = _real_data_evaluation(graph, accepted, inferred)
    graph_stats = {
        **graph.graph_stats(),
        "direct_edges": len(graph.edges),
        "inferred_edges": len(inferred),
        "inference_limits": {
            "maximum_facts": 5_000,
            "maximum_depth": 2,
            "source_graph_mutated": False,
        },
    }
    _json(output / "graph_stats.json", graph_stats)
    _json(output / "relations.json", {"relations": relations})
    _json(output / "relation_rules.json", {key: value for key, value in discovery.items() if key != "pair_metrics"})
    _json(output / "inferred_facts.json", facts_document(inferred))
    _json(output / "proofs.json", facts_document({**graph.direct, **inferred}))
    _json(output / "evaluation.json", evaluation)
    _visualization(graph, inferred, discovery, relations, evaluation, output / "ontology.html")
    return {
        "stats": graph_stats, "discovery": discovery, "evaluation": evaluation,
        "output": output, "inferred": inferred,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.database, args.output)
    stats, discovery = result["stats"], result["discovery"]
    print("=== V682 REAL GRAPH ===")
    print(f"database: {stats['database']}")
    print(f"entities: {stats['entities']}")
    print(f"edges: {stats['edges']}")
    print(f"unique relations: {stats['unique_relations']}")
    print(f"tables used: {', '.join(stats['tables_used'])}")
    print(f"rows examined: {stats['rows_examined']}")
    print("\n=== RULE DISCOVERY ===")
    print(f"candidate rules: {discovery['candidate_rules']}")
    print(f"accepted: {discovery['accepted_rules']}")
    print(f"rejected: {discovery['rejected_rules']}")
    print("top discovered rules:")
    for rule in discovery["rules"][:5]:
        print(f"  {rule['left_relation']} + {rule['right_relation']} -> {rule['result_relation']}"
              f" | support={rule['support']} precision={rule['precision']} confidence={rule['confidence']}"
              f" [{rule['status']}]")
    print("\n=== INFERENCE ===")
    print(f"direct facts: {stats['direct_edges']}")
    print(f"inferred facts: {stats['inferred_edges']}")
    print("\n=== VISUALIZATION ===")
    print(f"ontology.html: {result['output'] / 'ontology.html'}")
    print("\n=== REAL-DATA TEST ===")
    for item in result["evaluation"]["original_dog_chain"]["results"]:
        print(f"{item['query']}: {item['status']} {item.get('evidence_kind') or ''}".rstrip())
    selected = result["evaluation"]["automatic_equivalent_chain"]
    if selected and selected.get("proof"):
        print("automatic chain:")
        print(_proof_text(selected["proof"]))


if __name__ == "__main__":
    main()
