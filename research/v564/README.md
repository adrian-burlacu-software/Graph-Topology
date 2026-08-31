# V564 — 50GB Semantic Graph Referential Audit

V564 fixes V563's database-opening problem and keeps the audit entirely
disk-backed.

The source graph is opened SQLite read-only.

The audit does NOT perform a global `edges JOIN edges` over the 192M-edge
database.

It samples subjects and retrieves their outgoing adjacency using the existing
subject index, then performs bounded two-hop composition on the sampled
neighborhoods.

The opener explicitly resolves Windows paths and prints the absolute database
path and size before accessing SQLite.

Run from the Graph-Topology repository root:

```powershell
python .\research\v564\v564_semantic_graph_audit.py --database ".\results\v561_kg_composition_audit.sqlite" --output ".\results\v564_semantic_graph_audit.json" --sample-subjects 20000 --windows 100 --batch-size 500 --max-paths-per-subject 100 --max-mid-edges 250 --min-paths 25 --top-relations 100 --top-rules 50 --output-rules 500 --lookup-queries 1000 --seed 564
```

If the database has another name, pass its exact path to `--database`.

The script will explicitly list nearby `.sqlite` files if the requested path
does not exist.
