# V542 — DB-Native Knowledge Verdict Discovery

This audit operates only on the long-term semantic graph. It excludes `live_facts`/conversation tables.

It discovers four epistemic classes from actual database content:

- `SUPPORTED`: direct positive fact.
- `INDIRECT-SUPPORTED`: safe, typed multi-hop composition.
- `REFUTED`: explicit negative predicate stored in the graph.
- `UNKNOWN`: hard contrast with neither positive nor explicit negative support.

No LLM or conversation state is used.

Default parallelism is 20.
