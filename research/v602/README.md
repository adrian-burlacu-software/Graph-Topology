# V602 — Global Conditional Attention + BFS Continuation

V602 is the architectural experiment following the global conditional-prior result.
The prior is trained globally from the SQLite graph and used as an **attention
allocator**, while the actual traversal is BFS.

```text
global conditional prior
        |
        v
attention-ranked frontier
        |
        v
      BFS
        |
        v
attention budget exhausted
        |
        v
BFS CONTINUES FROM THE SAME FRONTIER
```

The target relation is terminal-only: it closes a multi-hop proof but is not
allowed to become an ordinary interior expansion. The controller therefore tests
whether learned attention can seed a useful frontier and let ordinary BFS finish
the computation.

The script is self-contained. Runtime dependencies are only Python + the SQLite
graph database. No V568/V595/V599/V600/V601/V602 artifact is loaded.

## Run

```powershell
python .\research\v602\v602_global_conditional_attention_bfs.py --database ".\results\v562_kg_composition_audit.sqlite" --output ".\results\v602_global_conditional_attention_bfs.json" --workers 20 --seeds 5 --seed-start 60200 --budget 80 --per-node 60 --max-depth 3 --cache-entries 12000 --max-probes-per-case 500 --train-cases-per-relation 120 --supported-per-relation 20 --negative-per-relation 20 --prior-decay 0.65 --attention-fraction 0.35 --exploration-quota 0.15 --progress-every 10
```

Key telemetry:

```text
attention_steps
bfs_steps
attention_selected
exploration_selected
depth_hist
target_hits
```

SQLite remains read-only.
