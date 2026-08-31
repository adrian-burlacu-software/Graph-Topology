# V593 — Goal-Directed Multi-Hop Proof Construction

V593 is the corrected compositional experiment.

The previous V592 run never reached depth 2. V593 makes depth 2 mandatory:
depth 1 generates intermediate nodes, and every retained intermediate is
expanded before a case can finish.

A positive proof must:
- contain at least 2 hops;
- end in the requested relation and object;
- have continuous edges;
- have every edge independently verified in SQLite.

Direct target edges are explicitly rejected as multi-hop proofs.

The case adapter lives INSIDE the policy, so raw oracle dictionaries are
normalized before any `case.gold` access.

## Run

```powershell
python .\research\v593\v593_goal_directed_multihop_proof.py --database ".\results\v562_kg_composition_audit.sqlite" --v568 ".\results\v568_20_relation_induction.json" --output ".\results\v593_goal_directed_multihop.json" --workers 20 --seeds 5 --seed-start 59300 --budget 80 --per-node 60 --max-depth 3 --cache-entries 12000 --progress-every 10 --max-probes-per-case 500 --max-case-seconds 2 --top-predicates 7 --sample-each 600 --middle-limit 150 --case-cap 100 --supported-per-relation 20 --negative-per-relation 20
```

Watch the telemetry:

```text
d1 > 0
d2 > 0
multi_hop_candidates > 0
verified > 0
```

A raw-dictionary two-hop smoke test is included in build validation.
