# V204B — Comprehensive SmolLM2 Teacher Qualification

This is a replacement for the first V204 teacher test.

The first test was not useful because:
- case construction could produce fewer cases than requested;
- the prompt was too brittle;
- empty-edge JSON technically parsed but was not a useful semantic answer.

V204B fixes that by testing the local SmolLM2-1.7B-Instruct model across four
teacher tasks in one run.

## Tasks

### A. Relation classification — 50 cases

Input:

```text
A dog is an animal.
```

The model must return:

```json
{"relation":"IsA"}
```

This measures whether the teacher can identify relation semantics reliably.

### B. Single-fact graph extraction — 50 cases

Input:

```text
A dog is an animal.
```

Expected:

```json
{
  "nodes": ["dog", "animal"],
  "edges": [
    {"source":"dog","relation":"IsA","target":"animal"}
  ]
}
```

### C. Multi-fact graph extraction — 30 cases

Input:

```text
A dog is an animal. A dog can bark.
```

Expected two edges.

### D. Cognitive state / trajectory extraction — 20 cases

Input describes a small state transition:

```text
The dog is active. The animal concept is not active yet.
The next state activates animal and binds dog to animal as IsA.
```

The model must return:

```json
{
  "current_nodes": [...],
  "next_nodes": [...],
  "edges_added": [...]
}
```

## One run

From `research/`:

```powershell
python .\v204b_teacher_qualification\qualify_teacher.py
```

Optional:

```powershell
python .\v204b_teacher_qualification\qualify_teacher.py --max-new-tokens 180
```

Expected exactly:

```text
50 relation cases
50 single graph cases
30 multi graph cases
20 trajectory cases
150 total
```

Outputs:

```text
results/v204b_teacher_qualification.json
results/v204b_teacher_responses.jsonl
```

## Interpretation

The teacher is useful if it can:

- reliably choose relations;
- produce graph edges with low hallucination;
- handle multiple facts;
- represent state transitions.

The teacher is *not* required to be perfect. The purpose is to determine whether
it is good enough to supply filtered/noisy supervision to the graph architecture.

Every raw response is cached so we can inspect failures without rerunning the
LLM.
