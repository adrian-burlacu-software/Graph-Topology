# V511 — Attention Trajectory Benchmark

V511 is an instrumentation release. It does **not** alter the knowledge base,
answering rules, or V509/V510 correctness criteria.

## Purpose

Measure the planner/attention trajectory when the conversational target is
stable versus when the target changes.

Each turn records:

- goal
- topic
- target signature
- whether the target changed from the previous turn
- planner candidates and component scores
- winning source and score
- winner margin
- a small switch-cost proxy

Knowledge should remain frozen during the benchmark.

## New probes

V511 adds four explicit target-switch probes:

1. dog property switch
2. subject switch
3. subject + property switch
4. return to a previous subject

The main V509/V510 suites remain unchanged so correctness can be compared
directly with V510.
