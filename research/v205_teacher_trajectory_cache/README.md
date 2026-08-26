# V205 — Validated SmolLM2 Teacher Trajectory Cache

V204B showed that SmolLM2 is too noisy to be a semantic graph authority, but it
was substantially better at structured trajectory/state prompts.

V205 uses the LLM only as a **planner/teacher**.

The semantic graph remains authoritative.

## Core idea

```text
ConceptNet long-term memory
        |
        v
working-memory state
        |
        v
generate VALID candidate actions
        |
        v
SmolLM2 chooses candidate action
        |
        v
validator
        |
   accept / reject
        |
        v
cached teacher trajectory
```

The teacher does not invent:

- relation IDs
- arbitrary node names
- graph edges
- action arguments

Instead, the script gives it a numbered list of **already-valid candidates**.
The model selects the candidate IDs it thinks are useful.

This dramatically reduces the opportunity for hallucination while preserving
the LLM's role as a noisy planner.

## Candidate actions

Candidates are constructed from the actual current graph:

```text
NOOP
REUSE(existing node)
BIND(existing semantic edge)
INHIBIT(active node)
BRANCH(existing node + existing relation)
CREATE
COMMIT
```

`BIND` is only offered when the corresponding semantic edge exists in
ConceptNet.

## Cached output

Every teacher request is stored in:

```text
results/v205_teacher_trajectories.jsonl
```

Each record contains:

- scenario
- working-memory graph
- candidate actions
- exact prompt
- raw SmolLM2 output
- parsed candidate IDs
- validation result
- accepted action
- resulting graph state
- teacher confidence, if emitted

A compact accepted dataset is also written:

```text
results/v205_teacher_dataset.jsonl
```

**Training the graph architecture never needs to invoke SmolLM2 again.**

## Run

From `research/`:

```powershell
python .\v205_teacher_trajectory_cache\generate_teacher_data.py
```

Default:

```text
500 teacher decisions
3 deterministic prompt variants
CUDA if available
cached locally
```

More:

```powershell
python .\v205_teacher_trajectory_cache\generate_teacher_data.py --cases 2000
```

Fewer:

```powershell
python .\v205_teacher_trajectory_cache\generate_teacher_data.py --cases 250
```

The script automatically resumes from an existing JSONL cache and skips cases
already generated.

## Output quality

The useful quantity is:

```text
acceptance_rate
```

not raw LLM accuracy.

The teacher can still be wrong. The validator guarantees that accepted actions
are structurally grounded in the actual semantic graph.

The graph architecture can therefore learn from a **large, cached, noisy but
validated teacher trajectory set** without paying LLM inference cost during
training.
